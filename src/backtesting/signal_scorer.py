"""
Unified Signal Scorer for MetaQuant Nigeria Backtesting.
Combines signals from ALL modules into a composite score.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalScorer:
    """
    Unified signal scoring combining all MetaQuant data sources.
    
    Signal Sources:
    - ML Predictions (25%): Direction, probability, confidence
    - ML Anomalies (15%): Volume spikes, smart money, accumulation
    - ML Clusters (10%): Cluster momentum, similar stocks
    - Flow Analysis (20%): Delta, VWAP, absorption
    - Fundamentals (15%): Valuation, P/E, growth
    - Market Intel (15%): Breadth, sector strength, regime
    
    Output: Score from -1.0 (strong sell) to +1.0 (strong buy)
    """
    
    # Default weights (customizable)
    DEFAULT_WEIGHTS = {
        'ml_prediction': 0.25,
        'ml_anomaly': 0.15,
        'ml_cluster': 0.10,
        'flow': 0.20,
        'fundamental': 0.15,
        'market_intel': 0.15
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the signal scorer.
        
        Args:
            weights: Optional custom weights for each signal source
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def score_ml_prediction(self, prediction: Dict) -> float:
        """
        Score from ML price prediction.
        
        Args:
            prediction: Dict with direction, probabilities, confidence
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not prediction or not prediction.get('success'):
            return 0.0
        
        direction = prediction.get('direction', 'FLAT')
        probs = prediction.get('probabilities', {})
        confidence = prediction.get('confidence', 50)
        
        # Base score from direction
        if direction == 'UP':
            base = 1.0
        elif direction == 'DOWN':
            base = -1.0
        else:
            base = 0.0
        
        # Weight by probability spread
        up_prob = probs.get('UP', 0.33)
        down_prob = probs.get('DOWN', 0.33)
        prob_factor = abs(up_prob - down_prob)  # 0 to ~0.66
        
        # Weight by confidence
        conf_factor = confidence / 100.0
        
        return base * prob_factor * conf_factor
    
    def score_ml_anomaly(self, anomalies: List[Dict]) -> float:
        """
        Score from ML anomaly detection.
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not anomalies:
            return 0.0
        
        score = 0.0
        
        for a in anomalies:
            atype = a.get('type', '')
            severity = a.get('severity', 50) / 100.0
            
            # Accumulation is bullish
            if atype == 'accumulation':
                score += severity * 0.5
            # Distribution is bearish
            elif atype == 'distribution':
                score -= severity * 0.5
            # Volume spike - neutral but adds conviction to direction
            elif atype == 'volume_spike':
                score += severity * 0.1
            # Smart money - follow the smart money
            elif atype in ['smart_money_buy', 'smart_money']:
                score += severity * 0.3
        
        return max(-1.0, min(1.0, score))
    
    def score_ml_cluster(self, cluster_info: Dict, cluster_momentum: Dict) -> float:
        """
        Score from ML cluster analysis.
        
        Args:
            cluster_info: Dict with cluster assignment
            cluster_momentum: Dict with cluster performance
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not cluster_info:
            return 0.0
        
        cluster_id = cluster_info.get('cluster')
        if cluster_id is None:
            return 0.0
        
        # Get cluster momentum
        momentum = cluster_momentum.get(cluster_id, 0)
        
        # Normalize momentum to [-1, 1]
        return max(-1.0, min(1.0, momentum / 5.0))
    
    def score_flow(self, flow_data: Dict) -> float:
        """
        Score from flow/order analysis.
        
        Args:
            flow_data: Dict with delta, VWAP, absorption info
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not flow_data:
            return 0.0
        
        score = 0.0
        
        # Delta (cumulative buying vs selling)
        delta = flow_data.get('session_delta', 0)
        if delta > 0:
            score += min(0.4, delta / 1000000)  # Normalize
        else:
            score += max(-0.4, delta / 1000000)
        
        # VWAP position (price above = bullish, below = bearish)
        vwap_position = flow_data.get('vwap_position', 'AT')
        if vwap_position == 'ABOVE':
            score += 0.3
        elif vwap_position == 'BELOW':
            score -= 0.3
        
        # Absorption (high volume, low price movement = accumulation)
        if flow_data.get('absorption_bullish'):
            score += 0.3
        elif flow_data.get('absorption_bearish'):
            score -= 0.3
        
        return max(-1.0, min(1.0, score))
    
    def score_fundamental(self, fundamentals: Dict) -> float:
        """
        Score from fundamental analysis.
        
        Args:
            fundamentals: Dict with valuation metrics
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not fundamentals:
            return 0.0
        
        score = 0.0
        
        # P/E relative to sector (lower = better value)
        pe = fundamentals.get('pe_ratio', 0)
        sector_pe = fundamentals.get('sector_avg_pe', pe)
        if pe > 0 and sector_pe > 0:
            pe_ratio = pe / sector_pe
            if pe_ratio < 0.7:  # Cheap
                score += 0.3
            elif pe_ratio > 1.3:  # Expensive
                score -= 0.2
        
        # Dividend yield
        div_yield = fundamentals.get('dividend_yield', 0) or 0
        if div_yield > 5:
            score += 0.2
        elif div_yield > 3:
            score += 0.1
        
        # Earnings growth
        eps_growth = fundamentals.get('eps_growth', 0) or 0
        if eps_growth > 20:
            score += 0.3
        elif eps_growth > 10:
            score += 0.15
        elif eps_growth < -10:
            score -= 0.2
        
        # ROE
        roe = fundamentals.get('roe', 0) or 0
        if roe > 20:
            score += 0.2
        elif roe > 15:
            score += 0.1
        
        return max(-1.0, min(1.0, score))
    
    def score_market_intel(self, market_data: Dict) -> float:
        """
        Score from market intelligence.
        
        Args:
            market_data: Dict with breadth, regime, sector strength
            
        Returns:
            Score from -1.0 to +1.0
        """
        if not market_data:
            return 0.0
        
        score = 0.0
        
        # Market regime
        regime = market_data.get('regime', 'UNKNOWN')
        if regime == 'BULLISH':
            score += 0.4
        elif regime == 'BEARISH':
            score -= 0.4
        elif regime == 'ACCUMULATION':
            score += 0.2
        elif regime == 'DISTRIBUTION':
            score -= 0.2
        
        # Breadth
        breadth = market_data.get('breadth', 0)
        if breadth > 0.6:  # More advancers
            score += 0.3
        elif breadth < 0.4:  # More decliners
            score -= 0.3
        
        # Sector strength (relative to market)
        sector_strength = market_data.get('sector_relative_strength', 0)
        score += max(-0.3, min(0.3, sector_strength / 100))
        
        return max(-1.0, min(1.0, score))
    
    def compute_composite_score(
        self,
        ml_prediction: Optional[Dict] = None,
        ml_anomalies: Optional[List[Dict]] = None,
        ml_cluster: Optional[Dict] = None,
        cluster_momentum: Optional[Dict] = None,
        flow_data: Optional[Dict] = None,
        fundamentals: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Compute weighted composite score from all signals.
        
        Returns:
            Dict with composite_score, component_scores, signal
        """
        # Score each component
        scores = {
            'ml_prediction': self.score_ml_prediction(ml_prediction or {}),
            'ml_anomaly': self.score_ml_anomaly(ml_anomalies or []),
            'ml_cluster': self.score_ml_cluster(ml_cluster or {}, cluster_momentum or {}),
            'flow': self.score_flow(flow_data or {}),
            'fundamental': self.score_fundamental(fundamentals or {}),
            'market_intel': self.score_market_intel(market_data or {})
        }
        
        # Weighted sum
        composite = sum(
            scores[k] * self.weights.get(k, 0) 
            for k in scores
        )
        
        # Generate signal
        if composite > 0.3:
            signal = 'BUY'
        elif composite < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'composite_score': round(composite, 4),
            'component_scores': scores,
            'signal': signal,
            'weights': self.weights,
            'timestamp': datetime.now().isoformat()
        }
    
    def score_for_backtest(
        self,
        date: str,
        symbol: str,
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a symbol on a historical date for backtesting.
        
        Args:
            date: Date string (YYYY-MM-DD)
            symbol: Stock symbol
            historical_data: Dict with all historical signals
            
        Returns:
            Composite score result
        """
        # Extract signals for this date/symbol
        ml_pred = historical_data.get('predictions', {}).get(date, {}).get(symbol, {})
        ml_anom = historical_data.get('anomalies', {}).get(date, {}).get(symbol, [])
        ml_clust = historical_data.get('clusters', {}).get(symbol, {})
        clust_mom = historical_data.get('cluster_momentum', {}).get(date, {})
        flow = historical_data.get('flow', {}).get(date, {}).get(symbol, {})
        fund = historical_data.get('fundamentals', {}).get(date, {}).get(symbol, {})
        mkt = historical_data.get('market', {}).get(date, {})
        
        return self.compute_composite_score(
            ml_prediction=ml_pred,
            ml_anomalies=ml_anom,
            ml_cluster=ml_clust,
            cluster_momentum=clust_mom,
            flow_data=flow,
            fundamentals=fund,
            market_data=mkt
        )
