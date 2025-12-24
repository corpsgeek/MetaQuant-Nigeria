"""
Anomaly Detection for MetaQuant Nigeria.
Uses Isolation Forest and statistical methods to detect unusual trading patterns.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Anomaly detection disabled.")


class AnomalyType(Enum):
    """Types of market anomalies."""
    VOLUME_SPIKE = "volume_spike"
    PRICE_JUMP = "price_jump"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAK = "correlation_break"
    SMART_MONEY = "smart_money"
    UNKNOWN = "unknown"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    symbol: str
    anomaly_type: AnomalyType
    severity: float  # 0-100 scale
    description: str
    detected_at: datetime
    data: Dict[str, Any]


class AnomalyDetector:
    """
    Detects unusual trading patterns using Isolation Forest and statistical methods.
    
    Features:
    - Volume spike detection
    - Price jump detection
    - Smart money accumulation/distribution patterns
    - Volatility regime changes
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.available = SKLEARN_AVAILABLE
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Thresholds
        self.volume_zscore_threshold = 2.5
        self.price_zscore_threshold = 2.0
        self.volatility_zscore_threshold = 2.0
        
        # Recent anomalies cache
        self.recent_anomalies: List[Anomaly] = []
        self._max_cache_size = 100
        
    def train_isolation_forest(self, df: pd.DataFrame) -> bool:
        """
        Train the Isolation Forest model on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            True if training successful
        """
        if not self.available:
            return False
        
        try:
            features = self._compute_anomaly_features(df)
            if features.empty or len(features) < 50:
                logger.warning("Insufficient data for Isolation Forest training")
                return False
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features.values)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Expect 5% anomalies
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_scaled)
            
            logger.info(f"Isolation Forest trained on {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Isolation Forest training failed: {e}")
            return False
    
    def _compute_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for anomaly detection."""
        if df is None or df.empty or len(df) < 20:
            return pd.DataFrame()
        
        try:
            features = pd.DataFrame(index=df.index)
            
            close = df['close'].astype(float)
            volume = df['volume'].astype(float).fillna(0)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # Volume features
            vol_sma = volume.rolling(window=20).mean()
            vol_std = volume.rolling(window=20).std()
            features['volume_zscore'] = (volume - vol_sma) / vol_std.replace(0, 1e-10)
            features['volume_ratio'] = volume / vol_sma.replace(0, 1e-10)
            
            # Price features
            returns = close.pct_change()
            ret_sma = returns.rolling(window=20).mean()
            ret_std = returns.rolling(window=20).std()
            features['return_zscore'] = (returns - ret_sma) / ret_std.replace(0, 1e-10)
            features['return_abs'] = abs(returns)
            
            # Volatility features
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            atr_std = atr.rolling(window=20).std()
            features['atr_zscore'] = (atr - atr.rolling(window=20).mean()) / atr_std.replace(0, 1e-10)
            
            # Range features
            daily_range = (high - low) / close
            range_sma = daily_range.rolling(window=20).mean()
            range_std = daily_range.rolling(window=20).std()
            features['range_zscore'] = (daily_range - range_sma) / range_std.replace(0, 1e-10)
            
            # Gap features
            gap = (df['open'].astype(float) - close.shift(1)) / close.shift(1)
            gap_std = gap.rolling(window=20).std()
            features['gap_zscore'] = gap / gap_std.replace(0, 1e-10)
            
            # Clean up
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing anomaly features: {e}")
            return pd.DataFrame()
    
    def detect_anomalies(self, df: pd.DataFrame, symbol: str) -> List[Anomaly]:
        """
        Detect anomalies in the given data.
        
        Args:
            df: Recent OHLCV data
            symbol: Stock symbol
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if df is None or df.empty or len(df) < 20:
            return anomalies
        
        try:
            # Get latest data point
            latest = df.iloc[-1]
            close = latest['close']
            volume = latest['volume']
            
            # Compute features for statistical detection
            features = self._compute_anomaly_features(df)
            if features.empty:
                return anomalies
            
            latest_features = features.iloc[-1]
            
            # ========== VOLUME SPIKE DETECTION ==========
            vol_zscore = latest_features.get('volume_zscore', 0)
            if abs(vol_zscore) > self.volume_zscore_threshold:
                severity = min(100, abs(vol_zscore) * 20)
                anomalies.append(Anomaly(
                    symbol=symbol,
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    description=f"Volume {vol_zscore:.1f}σ {'above' if vol_zscore > 0 else 'below'} average",
                    detected_at=datetime.now(),
                    data={
                        'volume': volume,
                        'zscore': vol_zscore,
                        'avg_volume': volume / (1 + vol_zscore) if vol_zscore != -1 else 0
                    }
                ))
            
            # ========== PRICE JUMP DETECTION ==========
            ret_zscore = latest_features.get('return_zscore', 0)
            if abs(ret_zscore) > self.price_zscore_threshold:
                severity = min(100, abs(ret_zscore) * 25)
                anomalies.append(Anomaly(
                    symbol=symbol,
                    anomaly_type=AnomalyType.PRICE_JUMP,
                    severity=severity,
                    description=f"Price move {ret_zscore:.1f}σ from normal",
                    detected_at=datetime.now(),
                    data={
                        'price': close,
                        'zscore': ret_zscore,
                        'return': latest_features.get('return_abs', 0) * 100
                    }
                ))
            
            # ========== VOLATILITY SPIKE ==========
            atr_zscore = latest_features.get('atr_zscore', 0)
            if abs(atr_zscore) > self.volatility_zscore_threshold:
                severity = min(100, abs(atr_zscore) * 25)
                anomalies.append(Anomaly(
                    symbol=symbol,
                    anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                    severity=severity,
                    description=f"Volatility {atr_zscore:.1f}σ {'above' if atr_zscore > 0 else 'below'} normal",
                    detected_at=datetime.now(),
                    data={
                        'atr_zscore': atr_zscore
                    }
                ))
            
            # ========== SMART MONEY DETECTION ==========
            # Low volatility + high volume = potential accumulation
            if vol_zscore > 2 and abs(ret_zscore) < 0.5:
                anomalies.append(Anomaly(
                    symbol=symbol,
                    anomaly_type=AnomalyType.ACCUMULATION,
                    severity=60,
                    description="High volume with stable price - possible accumulation",
                    detected_at=datetime.now(),
                    data={
                        'volume_zscore': vol_zscore,
                        'price_zscore': ret_zscore
                    }
                ))
            
            # High volume + sharp drop = distribution
            if vol_zscore > 2 and ret_zscore < -1.5:
                anomalies.append(Anomaly(
                    symbol=symbol,
                    anomaly_type=AnomalyType.DISTRIBUTION,
                    severity=70,
                    description="High volume selloff - distribution pattern",
                    detected_at=datetime.now(),
                    data={
                        'volume_zscore': vol_zscore,
                        'price_zscore': ret_zscore
                    }
                ))
            
            # ========== ISOLATION FOREST DETECTION ==========
            if self.available and self.isolation_forest is not None and self.scaler is not None:
                try:
                    X = features.iloc[-1:].values
                    X_scaled = self.scaler.transform(X)
                    prediction = self.isolation_forest.predict(X_scaled)[0]
                    score = self.isolation_forest.score_samples(X_scaled)[0]
                    
                    if prediction == -1:  # Anomaly detected
                        severity = min(100, abs(score) * 50)
                        anomalies.append(Anomaly(
                            symbol=symbol,
                            anomaly_type=AnomalyType.SMART_MONEY,
                            severity=severity,
                            description=f"ML-detected unusual pattern (score: {score:.3f})",
                            detected_at=datetime.now(),
                            data={
                                'isolation_score': score,
                                'features': latest_features.to_dict()
                            }
                        ))
                except Exception as e:
                    logger.debug(f"Isolation Forest prediction failed: {e}")
            
            # Cache anomalies
            for a in anomalies:
                self._add_to_cache(a)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def detect_accumulation_distribution(self, df: pd.DataFrame, symbol: str, lookback: int = 20) -> Dict[str, Any]:
        """
        Detect accumulation/distribution patterns over a period.
        
        Args:
            df: OHLCV data
            symbol: Stock symbol
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with accumulation/distribution analysis
        """
        if df is None or df.empty or len(df) < lookback:
            return {'pattern': 'UNKNOWN', 'confidence': 0}
        
        try:
            recent = df.tail(lookback)
            
            close = recent['close'].astype(float)
            volume = recent['volume'].astype(float)
            high = recent['high'].astype(float)
            low = recent['low'].astype(float)
            
            # Calculate Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # A/D Line
            ad_line = mfv.cumsum()
            
            # Price trend
            price_trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            
            # AD trend
            ad_trend = (ad_line.iloc[-1] - ad_line.iloc[0])
            
            # Determine pattern
            if price_trend > 0.02 and ad_trend > 0:
                pattern = "STRONG_ACCUMULATION"
                description = "Price up with positive money flow"
                confidence = min(100, abs(price_trend) * 500 + abs(ad_trend) / volume.mean())
            elif price_trend < -0.02 and ad_trend < 0:
                pattern = "DISTRIBUTION"
                description = "Price down with negative money flow"
                confidence = min(100, abs(price_trend) * 500 + abs(ad_trend) / volume.mean())
            elif price_trend < 0 and ad_trend > 0:
                pattern = "ACCUMULATION"
                description = "Price stable/down but money flowing in"
                confidence = 70
            elif price_trend > 0 and ad_trend < 0:
                pattern = "DISTRIBUTION"
                description = "Price up but money flowing out"
                confidence = 70
            else:
                pattern = "NEUTRAL"
                description = "No clear accumulation/distribution"
                confidence = 30
            
            return {
                'pattern': pattern,
                'description': description,
                'confidence': confidence,
                'price_change': price_trend * 100,
                'ad_change': ad_trend,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"A/D analysis failed: {e}")
            return {'pattern': 'UNKNOWN', 'confidence': 0}
    
    def get_recent_anomalies(self, symbol: Optional[str] = None, limit: int = 20) -> List[Anomaly]:
        """Get recently detected anomalies."""
        if symbol:
            return [a for a in self.recent_anomalies if a.symbol == symbol][:limit]
        return self.recent_anomalies[:limit]
    
    def _add_to_cache(self, anomaly: Anomaly):
        """Add anomaly to cache."""
        self.recent_anomalies.insert(0, anomaly)
        if len(self.recent_anomalies) > self._max_cache_size:
            self.recent_anomalies = self.recent_anomalies[:self._max_cache_size]
    
    def save_model(self, name: str = "default"):
        """Save trained model to disk."""
        if self.isolation_forest is None:
            return
        
        try:
            model_path = self.model_dir / f"anomaly_{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'isolation_forest': self.isolation_forest,
                    'scaler': self.scaler
                }, f)
            logger.info(f"Anomaly model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, name: str = "default") -> bool:
        """Load trained model from disk."""
        try:
            model_path = self.model_dir / f"anomaly_{name}.pkl"
            if not model_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.isolation_forest = data['isolation_forest']
            self.scaler = data['scaler']
            
            logger.info(f"Anomaly model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
