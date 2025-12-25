"""
LSTM Neural Network Predictor for MetaQuant Nigeria.
Deep learning model for stock price direction and magnitude prediction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM prediction disabled.")


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences."""
    
    def __init__(self, X: np.ndarray, y_dir: np.ndarray, y_mag: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_dir = torch.LongTensor(y_dir)
        self.y_mag = torch.FloatTensor(y_mag)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_dir[idx], self.y_mag[idx]


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for stock prediction.
    
    Architecture:
    - 2 LSTM layers with dropout
    - Dense layers for classification and regression
    - Dual output: direction (3-class) + magnitude (regression)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Direction classifier (3 classes: DOWN, FLAT, UP)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
        
        # Magnitude regressor
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        
        # Direction prediction
        direction = self.direction_head(last_output)
        
        # Magnitude prediction
        magnitude = self.magnitude_head(last_output).squeeze(-1)
        
        return direction, magnitude


class LSTMPredictor:
    """
    LSTM-based predictor for stock price direction and magnitude.
    
    Features:
    - Uses 30-day sliding windows of technical features
    - Dual output: classification + regression
    - GPU acceleration when available
    - Model persistence for fast inference
    """
    
    # Prediction thresholds
    UP_THRESHOLD = 0.01
    DOWN_THRESHOLD = -0.01
    
    def __init__(self, model_dir: Optional[str] = None, sequence_length: int = 30):
        self.available = TORCH_AVAILABLE
        self.sequence_length = sequence_length
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        # Model components
        self.model: Optional[LSTMModel] = None
        self.feature_columns: List[str] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Training metadata
        self.trained_symbol: Optional[str] = None
        self.training_date: Optional[datetime] = None
        self.training_accuracy: float = 0.0
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicator features from OHLCV data."""
        if df is None or df.empty or len(df) < 30:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            volume = df['volume'].astype(float).fillna(0)
            
            # Price-based features (normalized)
            features['return_1d'] = close.pct_change() * 100
            features['return_5d'] = close.pct_change(5) * 100
            features['return_10d'] = close.pct_change(10) * 100
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands position
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            features['bb_position'] = (close - sma20) / (2 * std20 + 1e-10)
            
            # ATR ratio
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            features['atr_ratio'] = tr.rolling(14).mean() / close
            
            # Volume features
            features['volume_ratio'] = volume / volume.rolling(20).mean().replace(0, 1e-10)
            
            # Price relative to MAs
            features['price_to_sma10'] = close / close.rolling(10).mean() - 1
            features['price_to_sma20'] = close / close.rolling(20).mean() - 1
            
            # Volatility
            features['volatility'] = close.pct_change().rolling(10).std() * np.sqrt(252)
            
            # Higher high / lower low
            features['higher_high'] = (high > high.shift(1)).astype(float)
            features['higher_low'] = (low > low.shift(1)).astype(float)
            
            # Clean up
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            
            self.feature_columns = features.columns.tolist()
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return pd.DataFrame()
    
    def prepare_sequences(self, features: pd.DataFrame, targets: pd.DataFrame = None) -> Tuple:
        """Prepare sliding window sequences for LSTM."""
        X_sequences = []
        y_dir_sequences = []
        y_mag_sequences = []
        
        values = features.values
        
        for i in range(len(values) - self.sequence_length):
            X_sequences.append(values[i:i + self.sequence_length])
            
            if targets is not None:
                y_dir_sequences.append(targets['direction'].iloc[i + self.sequence_length])
                y_mag_sequences.append(targets['magnitude'].iloc[i + self.sequence_length])
        
        X = np.array(X_sequences)
        y_dir = np.array(y_dir_sequences) if targets is not None else None
        y_mag = np.array(y_mag_sequences) if targets is not None else None
        
        return X, y_dir, y_mag
    
    def train(self, df: pd.DataFrame, symbol: str, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the LSTM model."""
        if not self.available:
            return {'success': False, 'error': 'PyTorch not available'}
        
        try:
            logger.info(f"Training LSTM model for {symbol}...")
            
            # Compute features
            features = self.compute_features(df)
            if features.empty or len(features) < self.sequence_length + 50:
                logger.warning(f"{symbol}: Insufficient data for LSTM - got {len(features)} rows")
                return {'success': False, 'error': f'Insufficient data ({len(features)} rows)'}
            
            # Normalize features
            self.feature_means = features.mean().values
            self.feature_stds = features.std().values
            self.feature_stds[self.feature_stds == 0] = 1  # Avoid division by zero
            
            normalized = (features.values - self.feature_means) / self.feature_stds
            normalized_df = pd.DataFrame(normalized, index=features.index, columns=features.columns)
            
            # Create targets
            close = df.loc[features.index, 'close'].astype(float)
            future_return = close.shift(-1) / close - 1
            
            targets = pd.DataFrame(index=features.index)
            targets['direction'] = 1  # Default FLAT
            targets.loc[future_return > self.UP_THRESHOLD, 'direction'] = 2  # UP
            targets.loc[future_return < self.DOWN_THRESHOLD, 'direction'] = 0  # DOWN
            targets['magnitude'] = future_return * 100
            
            # Prepare sequences
            X, y_dir, y_mag = self.prepare_sequences(normalized_df, targets)
            
            if len(X) < 50:
                return {'success': False, 'error': 'Insufficient sequences for training'}
            
            # Train/test split (time-series)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
            y_mag_train, y_mag_test = y_mag[:split_idx], y_mag[split_idx:]
            
            # Create datasets
            train_dataset = StockDataset(X_train, y_dir_train, y_mag_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            input_size = X.shape[2]
            self.model = LSTMModel(input_size=input_size).to(self.device)
            
            # Loss functions
            dir_criterion = nn.CrossEntropyLoss()
            mag_criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_dir_batch, y_mag_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_dir_batch = y_dir_batch.to(self.device)
                    y_mag_batch = y_mag_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    dir_pred, mag_pred = self.model(X_batch)
                    
                    dir_loss = dir_criterion(dir_pred, y_dir_batch)
                    mag_loss = mag_criterion(mag_pred, y_mag_batch)
                    loss = dir_loss + 0.1 * mag_loss
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Evaluate
            self.model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(self.device)
                dir_pred, mag_pred = self.model(X_test_t)
                dir_pred_class = dir_pred.argmax(dim=1).cpu().numpy()
                accuracy = (dir_pred_class == y_dir_test).mean()
            
            # Save metadata
            self.trained_symbol = symbol
            self.training_date = datetime.now()
            self.training_accuracy = accuracy
            
            # Save model
            self._save_model(symbol)
            
            logger.info(f"LSTM training complete for {symbol}: accuracy={accuracy:.2%}")
            
            return {
                'success': True,
                'symbol': symbol,
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'epochs': epochs
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Make prediction using LSTM model."""
        if not self.available:
            return {'success': False, 'error': 'PyTorch not available'}
        
        # Load model if needed
        if self.model is None or self.trained_symbol != symbol:
            if not self._load_model(symbol):
                # Train if no model exists
                train_result = self.train(df, symbol)
                if not train_result.get('success'):
                    return {'success': False, 'error': 'Could not train LSTM model'}
        
        try:
            # Compute and normalize features
            features = self.compute_features(df)
            if features.empty or len(features) < self.sequence_length:
                return {'success': False, 'error': 'Could not compute features'}
            
            normalized = (features.values - self.feature_means) / self.feature_stds
            
            # Get last sequence
            X = normalized[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                dir_pred, mag_pred = self.model(X_tensor)
                dir_proba = torch.softmax(dir_pred, dim=1).cpu().numpy()[0]
                dir_class = dir_pred.argmax(dim=1).item()
                magnitude = mag_pred.item()
            
            # Map direction
            direction_labels = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
            direction = direction_labels.get(dir_class, 'UNKNOWN')
            
            # Confidence
            confidence = max(dir_proba) * 100
            
            # Current price
            current_price = float(df['close'].iloc[-1])
            predicted_price = current_price * (1 + magnitude / 100)
            
            return {
                'success': True,
                'symbol': symbol,
                'direction': direction,
                'direction_code': dir_class - 1,  # Map to -1, 0, 1
                'confidence': confidence,
                'expected_return': magnitude,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'model_accuracy': self.training_accuracy * 100,
                'model': 'LSTM',
                'probabilities': {
                    'down': float(dir_proba[0]),
                    'flat': float(dir_proba[1]),
                    'up': float(dir_proba[2])
                }
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_model(self, symbol: str):
        """Save trained model to disk."""
        try:
            model_path = self.model_dir / f"lstm_{symbol.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model_state': self.model.state_dict(),
                    'input_size': self.model.lstm.input_size,
                    'feature_columns': self.feature_columns,
                    'feature_means': self.feature_means,
                    'feature_stds': self.feature_stds,
                    'trained_symbol': self.trained_symbol,
                    'training_date': self.training_date,
                    'training_accuracy': self.training_accuracy,
                    'sequence_length': self.sequence_length
                }, f)
            logger.info(f"LSTM model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")
    
    def _load_model(self, symbol: str) -> bool:
        """Load trained model from disk."""
        try:
            model_path = self.model_dir / f"lstm_{symbol.lower()}.pkl"
            if not model_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Recreate model
            input_size = data['input_size']
            self.model = LSTMModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(data['model_state'])
            self.model.eval()
            
            self.feature_columns = data['feature_columns']
            self.feature_means = data['feature_means']
            self.feature_stds = data['feature_stds']
            self.trained_symbol = data['trained_symbol']
            self.training_date = data['training_date']
            self.training_accuracy = data['training_accuracy']
            self.sequence_length = data['sequence_length']
            
            logger.info(f"LSTM model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False
