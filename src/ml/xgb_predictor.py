"""
XGBoost-based Price Predictor for MetaQuant Nigeria.
Uses technical indicators as features to predict price direction and magnitude.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost/sklearn not available. Price prediction disabled.")


class XGBPredictor:
    """
    XGBoost-based predictor for stock price direction and returns.
    
    Features:
    - Predicts next-day price direction (UP/DOWN/FLAT)
    - Predicts expected return magnitude
    - Uses 40+ technical indicators as features
    - Saves/loads trained models to disk
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the XGBoost predictor.
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.available = XGB_AVAILABLE
        
        if model_dir is None:
            # Default to src/ml/models
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.direction_model: Optional[xgb.XGBClassifier] = None
        self.magnitude_model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Feature columns
        self.feature_columns: List[str] = []
        
        # Training metadata
        self.trained_symbol: Optional[str] = None
        self.training_date: Optional[datetime] = None
        self.training_accuracy: float = 0.0
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicator features from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            DataFrame with computed features
        """
        if df is None or df.empty or len(df) < 30:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        try:
            # Price columns
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            open_price = df['open'].astype(float)
            volume = df['volume'].astype(float).fillna(0)
            
            # ========== MOMENTUM INDICATORS ==========
            # RSI (multiple periods)
            for period in [7, 14, 21]:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss.replace(0, 1e-10)
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # Stochastic
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
            
            # ========== TREND INDICATORS ==========
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                features[f'sma_{period}'] = close.rolling(window=period).mean()
                features[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            
            # MA Crossovers (as ratios)
            features['sma_5_20_ratio'] = features['sma_5'] / features['sma_20'].replace(0, 1e-10)
            features['sma_10_50_ratio'] = features['sma_10'] / features['sma_50'].replace(0, 1e-10)
            features['sma_20_200_ratio'] = features['sma_20'] / features['sma_200'].replace(0, 1e-10)
            
            # Price relative to MAs
            features['price_to_sma_20'] = close / features['sma_20'].replace(0, 1e-10)
            features['price_to_sma_50'] = close / features['sma_50'].replace(0, 1e-10)
            
            # ========== VOLATILITY INDICATORS ==========
            # Bollinger Bands
            bb_sma = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            features['bb_upper'] = bb_sma + (bb_std * 2)
            features['bb_lower'] = bb_sma - (bb_std * 2)
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_sma
            
            # ATR
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(window=14).mean()
            features['atr_ratio'] = features['atr_14'] / close
            
            # ========== VOLUME INDICATORS ==========
            # Volume ratios
            features['volume_sma_20'] = volume.rolling(window=20).mean()
            features['volume_ratio'] = volume / features['volume_sma_20'].replace(0, 1e-10)
            
            # OBV
            obv = ((close > close.shift(1)).astype(int) - (close < close.shift(1)).astype(int)) * volume
            features['obv'] = obv.cumsum()
            features['obv_sma_10'] = features['obv'].rolling(window=10).mean()
            features['obv_momentum'] = features['obv'] / features['obv_sma_10'].replace(0, 1e-10)
            
            # ========== PRICE MOMENTUM ==========
            # Returns over various periods
            for period in [1, 3, 5, 10, 20]:
                features[f'return_{period}d'] = close.pct_change(periods=period) * 100
            
            # Price range
            features['daily_range'] = (high - low) / close * 100
            features['gap'] = (open_price - close.shift(1)) / close.shift(1) * 100
            
            # ========== CANDLESTICK PATTERNS (simplified) ==========
            features['body_ratio'] = abs(close - open_price) / (high - low + 1e-10)
            features['upper_shadow'] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / (high - low + 1e-10)
            features['lower_shadow'] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / (high - low + 1e-10)
            
            # ========== ADX (Average Directional Index) ==========
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            tr_14 = tr.rolling(window=14).sum()
            plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr_14.replace(0, 1e-10))
            minus_di = 100 * (minus_dm.rolling(window=14).sum() / tr_14.replace(0, 1e-10))
            
            features['plus_di'] = plus_di
            features['minus_di'] = minus_di
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            features['adx'] = dx.rolling(window=14).mean()
            
            # Clean up
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            
            # Store feature columns
            self.feature_columns = features.columns.tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return pd.DataFrame()
    
    def train(self, df: pd.DataFrame, symbol: str, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the XGBoost models on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for model identification
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        if not self.available:
            return {'success': False, 'error': 'XGBoost not available'}
        
        try:
            logger.info(f"Training XGBoost model for {symbol}...")
            
            # Compute features
            features = self.compute_features(df)
            if features.empty or len(features) < 50:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Align features with price data
            aligned_df = df.loc[features.index].copy()
            
            # Create targets
            # Direction: 1 = UP (>0.5%), -1 = DOWN (<-0.5%), 0 = FLAT
            future_return = aligned_df['close'].shift(-1) / aligned_df['close'] - 1
            direction = pd.Series(0, index=future_return.index)
            direction[future_return > 0.005] = 1
            direction[future_return < -0.005] = -1
            
            # Remove last row (no target)
            X = features.iloc[:-1].values
            y_direction = direction.iloc[:-1].values
            y_magnitude = (future_return.iloc[:-1] * 100).values  # Percentage return
            
            # Split data
            X_train, X_test, y_dir_train, y_dir_test, y_mag_train, y_mag_test = train_test_split(
                X, y_direction, y_magnitude, test_size=test_size, shuffle=False
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train direction classifier
            self.direction_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=3,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )
            
            # Remap labels for XGBoost (needs 0, 1, 2 not -1, 0, 1)
            y_dir_train_mapped = y_dir_train + 1
            y_dir_test_mapped = y_dir_test + 1
            
            self.direction_model.fit(X_train_scaled, y_dir_train_mapped)
            
            # Train magnitude regressor
            self.magnitude_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42
            )
            self.magnitude_model.fit(X_train_scaled, y_mag_train)
            
            # Evaluate
            dir_pred = self.direction_model.predict(X_test_scaled)
            dir_accuracy = accuracy_score(y_dir_test_mapped, dir_pred)
            
            mag_pred = self.magnitude_model.predict(X_test_scaled)
            mag_rmse = np.sqrt(mean_squared_error(y_mag_test, mag_pred))
            
            # Store metadata
            self.trained_symbol = symbol
            self.training_date = datetime.now()
            self.training_accuracy = dir_accuracy
            
            # Save model
            self._save_model(symbol)
            
            logger.info(f"XGBoost training complete for {symbol}: accuracy={dir_accuracy:.2%}, RMSE={mag_rmse:.2f}%")
            
            return {
                'success': True,
                'symbol': symbol,
                'direction_accuracy': dir_accuracy,
                'magnitude_rmse': mag_rmse,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Make predictions for a stock.
        
        Args:
            df: Recent OHLCV data
            symbol: Stock symbol
            
        Returns:
            Dictionary with predictions
        """
        if not self.available:
            return {'success': False, 'error': 'XGBoost not available'}
        
        # Try to load model if not trained
        if self.direction_model is None or self.trained_symbol != symbol:
            if not self._load_model(symbol):
                # Need to train first
                train_result = self.train(df, symbol)
                if not train_result.get('success'):
                    return {'success': False, 'error': 'Could not train model'}
        
        try:
            # Compute features
            features = self.compute_features(df)
            if features.empty:
                return {'success': False, 'error': 'Could not compute features'}
            
            # Get latest features
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Predict direction
            dir_pred_mapped = self.direction_model.predict(X_scaled)[0]
            dir_pred = int(dir_pred_mapped) - 1  # Map back to -1, 0, 1
            dir_proba = self.direction_model.predict_proba(X_scaled)[0]
            
            # Predict magnitude
            mag_pred = self.magnitude_model.predict(X_scaled)[0]
            
            # Get direction label
            direction_labels = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}
            direction = direction_labels.get(dir_pred, 'UNKNOWN')
            
            # Calculate confidence
            confidence = max(dir_proba) * 100
            
            # Current price
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + mag_pred / 100)
            
            return {
                'success': True,
                'symbol': symbol,
                'direction': direction,
                'direction_code': dir_pred,
                'confidence': confidence,
                'expected_return': mag_pred,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'model_accuracy': self.training_accuracy * 100,
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.direction_model is None:
            return {}
        
        try:
            importance = self.direction_model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        except:
            return {}
    
    def _save_model(self, symbol: str):
        """Save trained model to disk."""
        try:
            model_path = self.model_dir / f"xgb_{symbol.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'direction_model': self.direction_model,
                    'magnitude_model': self.magnitude_model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'trained_symbol': self.trained_symbol,
                    'training_date': self.training_date,
                    'training_accuracy': self.training_accuracy
                }, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self, symbol: str) -> bool:
        """Load trained model from disk."""
        try:
            model_path = self.model_dir / f"xgb_{symbol.lower()}.pkl"
            if not model_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.direction_model = data['direction_model']
            self.magnitude_model = data['magnitude_model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.trained_symbol = data['trained_symbol']
            self.training_date = data['training_date']
            self.training_accuracy = data['training_accuracy']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
