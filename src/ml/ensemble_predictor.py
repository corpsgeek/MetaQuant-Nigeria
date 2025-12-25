"""
Ensemble Predictor for MetaQuant Nigeria.
Combines XGBoost, LSTM, and LightGBM predictions for improved accuracy.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import base predictors
from .xgb_predictor import XGBPredictor

# Try to import LSTM
try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("LSTM predictor not available")

# Try to import LightGBM
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


class LightGBMPredictor:
    """LightGBM-based predictor (lighter and faster than XGBoost)."""
    
    UP_THRESHOLD = 0.01
    DOWN_THRESHOLD = -0.01
    
    def __init__(self, model_dir: Optional[str] = None):
        self.available = LGBM_AVAILABLE
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.direction_model = None
        self.magnitude_model = None
        self.feature_columns: List[str] = []
        self.trained_symbol: Optional[str] = None
        self.training_accuracy: float = 0.0
    
    def train(self, df: pd.DataFrame, symbol: str, features: pd.DataFrame) -> Dict[str, Any]:
        """Train LightGBM model using pre-computed features."""
        if not self.available:
            return {'success': False, 'error': 'LightGBM not available'}
        
        try:
            if features.empty or len(features) < 50:
                return {'success': False, 'error': 'Insufficient data'}
            
            # Align with price data
            aligned_df = df.loc[features.index].copy()
            
            # Create targets
            future_return = aligned_df['close'].shift(-1) / aligned_df['close'] - 1
            direction = pd.Series(1, index=future_return.index)  # FLAT
            direction[future_return > self.UP_THRESHOLD] = 2  # UP
            direction[future_return < self.DOWN_THRESHOLD] = 0  # DOWN
            
            X = features.iloc[:-1].values
            y_dir = direction.iloc[:-1].values
            y_mag = (future_return.iloc[:-1] * 100).values
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
            y_mag_train, y_mag_test = y_mag[:split_idx], y_mag[split_idx:]
            
            # Train direction classifier
            train_data = lgb.Dataset(X_train, label=y_dir_train)
            valid_data = lgb.Dataset(X_test, label=y_dir_test, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            self.direction_model = lgb.train(
                params, train_data,
                num_boost_round=200,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Train magnitude regressor
            train_data_mag = lgb.Dataset(X_train, label=y_mag_train)
            valid_data_mag = lgb.Dataset(X_test, label=y_mag_test, reference=train_data_mag)
            
            params_mag = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1
            }
            
            self.magnitude_model = lgb.train(
                params_mag, train_data_mag,
                num_boost_round=200,
                valid_sets=[valid_data_mag],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = self.direction_model.predict(X_test).argmax(axis=1)
            accuracy = (y_pred == y_dir_test).mean()
            
            self.trained_symbol = symbol
            self.training_accuracy = accuracy
            self.feature_columns = features.columns.tolist()
            
            # Save
            self._save_model(symbol)
            
            return {'success': True, 'accuracy': accuracy}
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, features: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Make prediction using LightGBM."""
        if not self.available or self.direction_model is None:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            X = features.iloc[-1:].values
            
            dir_proba = self.direction_model.predict(X)[0]
            dir_class = dir_proba.argmax()
            magnitude = self.magnitude_model.predict(X)[0]
            
            direction_labels = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
            
            return {
                'success': True,
                'direction': direction_labels[dir_class],
                'direction_code': dir_class - 1,
                'confidence': max(dir_proba) * 100,
                'expected_return': magnitude,
                'predicted_price': current_price * (1 + magnitude / 100),
                'model': 'LightGBM',
                'probabilities': {
                    'down': float(dir_proba[0]),
                    'flat': float(dir_proba[1]),
                    'up': float(dir_proba[2])
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _save_model(self, symbol: str):
        model_path = self.model_dir / f"lgbm_{symbol.lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'direction_model': self.direction_model,
                'magnitude_model': self.magnitude_model,
                'feature_columns': self.feature_columns,
                'trained_symbol': self.trained_symbol,
                'training_accuracy': self.training_accuracy
            }, f)
    
    def _load_model(self, symbol: str) -> bool:
        model_path = self.model_dir / f"lgbm_{symbol.lower()}.pkl"
        if not model_path.exists():
            return False
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.direction_model = data['direction_model']
            self.magnitude_model = data['magnitude_model']
            self.feature_columns = data['feature_columns']
            self.trained_symbol = data['trained_symbol']
            self.training_accuracy = data['training_accuracy']
            return True
        except:
            return False


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple ML models.
    
    Models:
    - XGBoost (gradient boosting, good for tabular data)
    - LSTM (recurrent network, good for sequences)
    - LightGBM (fast gradient boosting)
    
    Combination strategies:
    - Weighted voting based on model confidence
    - Meta-learning with stacking
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            self.model_dir = Path(__file__).parent / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictors
        self.xgb_predictor = XGBPredictor(model_dir=str(self.model_dir))
        self.lgbm_predictor = LightGBMPredictor(model_dir=str(self.model_dir)) if LGBM_AVAILABLE else None
        self.lstm_predictor = LSTMPredictor(model_dir=str(self.model_dir)) if LSTM_AVAILABLE else None
        
        # Model weights (learned from performance)
        self.weights = {
            'xgb': 0.4,
            'lstm': 0.35,
            'lgbm': 0.25
        }
        
        # Track which models are available
        self.available_models = ['xgb']
        if LSTM_AVAILABLE:
            self.available_models.append('lstm')
        if LGBM_AVAILABLE:
            self.available_models.append('lgbm')
        
        logger.info(f"Ensemble initialized with models: {self.available_models}")
    
    def train(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train all ensemble models."""
        results = {'symbol': symbol, 'models': {}}
        
        # Train XGBoost
        xgb_result = self.xgb_predictor.train(df, symbol)
        results['models']['xgb'] = xgb_result
        
        # Get features from XGBoost for other models
        features = self.xgb_predictor.compute_features(df, symbol)
        
        # Train LightGBM
        if self.lgbm_predictor:
            lgbm_result = self.lgbm_predictor.train(df, symbol, features)
            results['models']['lgbm'] = lgbm_result
        
        # Train LSTM
        if self.lstm_predictor:
            lstm_result = self.lstm_predictor.train(df, symbol)
            results['models']['lstm'] = lstm_result
        
        # Calculate success
        successful = sum(1 for r in results['models'].values() if r.get('success'))
        results['success'] = successful > 0
        results['models_trained'] = successful
        
        # Update weights based on accuracy
        self._update_weights(results['models'])
        
        return results
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Make ensemble prediction."""
        predictions = {}
        
        # Get XGBoost prediction
        xgb_pred = self.xgb_predictor.predict(df, symbol)
        if xgb_pred.get('success'):
            predictions['xgb'] = xgb_pred
        
        # Get LSTM prediction
        if self.lstm_predictor:
            lstm_pred = self.lstm_predictor.predict(df, symbol)
            if lstm_pred.get('success'):
                predictions['lstm'] = lstm_pred
        
        # Get LightGBM prediction
        if self.lgbm_predictor and self.lgbm_predictor._load_model(symbol):
            features = self.xgb_predictor.compute_features(df, symbol)
            if not features.empty:
                current_price = float(df['close'].iloc[-1])
                lgbm_pred = self.lgbm_predictor.predict(features, current_price)
                if lgbm_pred.get('success'):
                    predictions['lgbm'] = lgbm_pred
        
        if not predictions:
            return {'success': False, 'error': 'No models available'}
        
        # Combine predictions
        return self._combine_predictions(predictions, symbol)
    
    def _combine_predictions(self, predictions: Dict[str, Dict], symbol: str) -> Dict[str, Any]:
        """Combine predictions using weighted voting."""
        total_weight = 0
        weighted_probs = {'down': 0, 'flat': 0, 'up': 0}
        weighted_return = 0
        confidences = []
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.33)
            probs = pred.get('probabilities', {'down': 0.33, 'flat': 0.34, 'up': 0.33})
            
            for key in weighted_probs:
                weighted_probs[key] += weight * probs.get(key, 0.33)
            
            weighted_return += weight * pred.get('expected_return', 0)
            confidences.append(pred.get('confidence', 50))
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for key in weighted_probs:
                weighted_probs[key] /= total_weight
            weighted_return /= total_weight
        
        # Determine direction
        max_prob_key = max(weighted_probs, key=weighted_probs.get)
        direction_map = {'down': 'DOWN', 'flat': 'FLAT', 'up': 'UP'}
        direction = direction_map[max_prob_key]
        
        # Overall confidence
        confidence = weighted_probs[max_prob_key] * 100
        
        # Current price from first prediction
        first_pred = list(predictions.values())[0]
        current_price = first_pred.get('current_price', 0)
        predicted_price = current_price * (1 + weighted_return / 100)
        
        return {
            'success': True,
            'symbol': symbol,
            'direction': direction,
            'direction_code': {'DOWN': -1, 'FLAT': 0, 'UP': 1}[direction],
            'confidence': confidence,
            'expected_return': weighted_return,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'model': 'Ensemble',
            'models_used': list(predictions.keys()),
            'probabilities': weighted_probs,
            'individual_predictions': {k: v.get('direction') for k, v in predictions.items()},
            'model_confidences': {k: v.get('confidence') for k, v in predictions.items()}
        }
    
    def _update_weights(self, results: Dict[str, Dict]):
        """Update model weights based on training accuracy."""
        accuracies = {}
        for model, result in results.items():
            if result.get('success'):
                acc = result.get('accuracy') or result.get('direction_accuracy', 0.5)
                accuracies[model] = acc
        
        if accuracies:
            total = sum(accuracies.values())
            if total > 0:
                for model, acc in accuracies.items():
                    self.weights[model] = acc / total
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status."""
        return {
            'available_models': self.available_models,
            'weights': self.weights,
            'xgb_available': True,
            'lstm_available': LSTM_AVAILABLE,
            'lgbm_available': LGBM_AVAILABLE
        }
