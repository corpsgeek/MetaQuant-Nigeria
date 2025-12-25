"""
PCA Factor Engine - Extract and analyze market factors from stock returns.

Provides:
- Multi-factor extraction (Market, Size, Value, Momentum, Volatility)
- Rolling PCA for regime detection
- Factor exposure calculation per stock
- Market regime detection (Risk-On, Risk-Off, Rotation)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PCAFactorEngine:
    """
    Extracts principal components from stock returns to identify market factors.
    
    Factors:
    - PC1 (Market): Overall market direction
    - PC2 (Size): Large-cap vs small-cap rotation
    - PC3 (Value): Value vs growth tilt
    - PC4 (Momentum): Momentum factor
    - PC5 (Volatility): High-vol vs low-vol clustering
    """
    
    FACTOR_NAMES = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
    N_COMPONENTS = 5
    ROLLING_WINDOW = 60  # days
    
    def __init__(self, n_components: int = 5):
        """Initialize PCA Factor Engine."""
        self.n_components = min(n_components, self.N_COMPONENTS)
        self.pca = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        # Stored data
        self._returns_matrix = None  # stocks × days
        self._symbols = []
        self._factor_loadings = None  # stocks × factors
        self._factor_returns = None  # days × factors
        self._variance_explained = None
        self._rolling_regimes = []
        
    def fit(self, price_data: Dict[str, pd.DataFrame], min_days: int = 90) -> 'PCAFactorEngine':
        """
        Fit PCA on historical stock returns.
        
        Args:
            price_data: Dict of symbol -> DataFrame with 'close' prices
            min_days: Minimum days of data required
            
        Returns:
            self for chaining
        """
        # Build returns matrix
        returns_dict = {}
        
        for symbol, df in price_data.items():
            if df.empty or len(df) < min_days:
                continue
            
            close = df['close'].astype(float)
            returns = close.pct_change().dropna()
            
            if len(returns) >= min_days - 1:
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 10:
            logger.warning(f"Not enough stocks for PCA: {len(returns_dict)}")
            return self
        
        # Align all returns to common dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(axis=0, thresh=len(returns_df.columns) * 0.8)
        returns_df = returns_df.fillna(0)
        
        if len(returns_df) < min_days - 1:
            logger.warning(f"Not enough common dates for PCA: {len(returns_df)}")
            return self
        
        self._returns_matrix = returns_df
        self._symbols = list(returns_df.columns)
        
        # Standardize returns (across time for each stock)
        scaled_returns = self.scaler.fit_transform(returns_df)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self._factor_returns = self.pca.fit_transform(scaled_returns)  # days × factors
        
        # Factor loadings (how each stock relates to each factor)
        # Transpose to get stocks × factors
        self._factor_loadings = self.pca.components_.T  # n_features × n_components
        
        # Store variance explained
        self._variance_explained = self.pca.explained_variance_ratio_
        
        self._is_fitted = True
        
        logger.info(f"PCA fitted on {len(self._symbols)} stocks, {len(returns_df)} days")
        logger.info(f"Variance explained: {[f'{v:.1%}' for v in self._variance_explained]}")
        
        return self
    
    def get_factor_exposures(self, symbol: str) -> Dict[str, float]:
        """
        Get a stock's exposure (loading) on each factor.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict of factor_name -> exposure value
        """
        if not self._is_fitted:
            return {name: 0.0 for name in self.FACTOR_NAMES[:self.n_components]}
        
        if symbol not in self._symbols:
            return {name: 0.0 for name in self.FACTOR_NAMES[:self.n_components]}
        
        idx = self._symbols.index(symbol)
        loadings = self._factor_loadings[idx]
        
        return {
            self.FACTOR_NAMES[i]: float(loadings[i])
            for i in range(self.n_components)
        }
    
    def get_all_exposures(self) -> pd.DataFrame:
        """
        Get factor exposures for all stocks.
        
        Returns:
            DataFrame with stocks as index, factors as columns
        """
        if not self._is_fitted:
            return pd.DataFrame()
        
        return pd.DataFrame(
            self._factor_loadings,
            index=self._symbols,
            columns=self.FACTOR_NAMES[:self.n_components]
        )
    
    def get_factor_returns(self) -> pd.DataFrame:
        """
        Get historical returns of each factor as ACTUAL PERCENTAGES.
        
        Factor return = weighted average of stock returns using factor loadings.
        
        Returns:
            DataFrame with dates as index, factors as columns (values are decimal returns)
        """
        if not self._is_fitted or self._returns_matrix is None:
            return pd.DataFrame()
        
        # Calculate actual factor returns by projecting stock returns onto factor loadings
        # factor_return_t = sum(stock_return_t * loading_stock) / sum(abs(loadings))
        
        factor_returns_list = []
        dates = self._returns_matrix.index
        
        for i in range(self.n_components):
            loadings = self._factor_loadings[:, i]  # loadings for this factor
            
            # Normalize loadings to get portfolio weights
            pos_loadings = np.maximum(loadings, 0)
            neg_loadings = np.maximum(-loadings, 0)
            
            # Long stocks with positive loadings, short stocks with negative loadings
            if pos_loadings.sum() > 0 and neg_loadings.sum() > 0:
                pos_weights = pos_loadings / pos_loadings.sum()
                neg_weights = neg_loadings / neg_loadings.sum()
                
                # Factor return = long portfolio - short portfolio
                factor_returns = (
                    (self._returns_matrix.values @ pos_weights) - 
                    (self._returns_matrix.values @ neg_weights)
                )
            else:
                # Just use signed loadings as weights
                weights = loadings / np.abs(loadings).sum() if np.abs(loadings).sum() > 0 else loadings
                factor_returns = self._returns_matrix.values @ weights
            
            factor_returns_list.append(factor_returns)
        
        # Combine into DataFrame
        factor_returns_df = pd.DataFrame(
            np.column_stack(factor_returns_list),
            index=dates,
            columns=self.FACTOR_NAMES[:self.n_components]
        )
        
        return factor_returns_df
    
    def get_variance_explained(self) -> Dict[str, float]:
        """Get variance explained by each factor."""
        if not self._is_fitted:
            return {}
        
        return {
            self.FACTOR_NAMES[i]: float(self._variance_explained[i])
            for i in range(self.n_components)
        }
    
    def get_market_regime(self, lookback_days: int = 20) -> Dict:
        """
        Detect current market regime based on factor behavior.
        
        Returns:
            Dict with regime info:
            - regime: "Risk-On", "Risk-Off", "Rotation"
            - confidence: 0-1
            - factor_signals: per-factor direction
        """
        if not self._is_fitted or self._factor_returns is None:
            return {'regime': 'Unknown', 'confidence': 0, 'factor_signals': {}}
        
        # Get recent factor returns
        recent = self._factor_returns[-lookback_days:]
        
        # Market factor (PC1) direction
        market_return = np.mean(recent[:, 0])
        market_vol = np.std(recent[:, 0])
        
        # Size factor (PC2) - positive = large-cap favored
        size_signal = np.mean(recent[:, 1]) if self.n_components > 1 else 0
        
        # Volatility clustering (PC5 or last component)
        vol_idx = min(4, self.n_components - 1)
        vol_signal = np.std(recent[:, vol_idx])
        
        # Determine regime
        if market_return > 0.01 and vol_signal < 0.05:
            regime = "Risk-On"
            confidence = min(1.0, abs(market_return) * 10)
        elif market_return < -0.01 or vol_signal > 0.08:
            regime = "Risk-Off"
            confidence = min(1.0, max(abs(market_return), vol_signal) * 5)
        else:
            regime = "Rotation"
            confidence = 0.5
        
        # Per-factor signals
        factor_signals = {}
        for i in range(self.n_components):
            factor_return = np.mean(recent[:, i])
            if factor_return > 0.005:
                factor_signals[self.FACTOR_NAMES[i]] = "Bullish"
            elif factor_return < -0.005:
                factor_signals[self.FACTOR_NAMES[i]] = "Bearish"
            else:
                factor_signals[self.FACTOR_NAMES[i]] = "Neutral"
        
        return {
            'regime': regime,
            'confidence': round(confidence, 2),
            'factor_signals': factor_signals,
            'market_return': round(float(market_return), 4),
            'volatility': round(float(market_vol), 4)
        }
    
    def calculate_alpha_score(self, symbol: str, raw_signal: float) -> float:
        """
        Calculate factor-adjusted alpha.
        
        Removes systematic factor exposure to find pure stock-specific signal.
        
        Args:
            symbol: Stock symbol
            raw_signal: Raw trading signal (-1 to +1)
            
        Returns:
            Factor-adjusted alpha score
        """
        if not self._is_fitted:
            return raw_signal
        
        exposures = self.get_factor_exposures(symbol)
        
        # Get recent factor returns
        if self._factor_returns is None or len(self._factor_returns) < 5:
            return raw_signal
        
        recent_factor_returns = np.mean(self._factor_returns[-5:], axis=0)
        
        # Calculate expected return from factors
        factor_contribution = sum(
            exposures.get(self.FACTOR_NAMES[i], 0) * recent_factor_returns[i]
            for i in range(self.n_components)
        )
        
        # Alpha = Raw signal - factor contribution
        # Clip to reasonable range
        alpha = raw_signal - factor_contribution * 0.1
        return max(-1, min(1, alpha))
    
    def calculate_factor_alignment(self, symbol: str) -> float:
        """
        Calculate how well aligned a stock is with current factor momentum.
        
        Stocks aligned with outperforming factors get higher scores.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Alignment score (-1 to +1)
        """
        if not self._is_fitted:
            return 0.0
        
        exposures = self.get_factor_exposures(symbol)
        regime = self.get_market_regime()
        
        # Calculate alignment
        alignment = 0.0
        for factor_name, signal in regime['factor_signals'].items():
            exposure = exposures.get(factor_name, 0)
            
            if signal == "Bullish":
                alignment += exposure * 0.3
            elif signal == "Bearish":
                alignment -= exposure * 0.3
        
        return max(-1, min(1, alignment))
    
    def rolling_fit(self, price_data: Dict[str, pd.DataFrame], 
                   window: int = None) -> List[Dict]:
        """
        Perform rolling PCA to track factor evolution over time.
        
        Args:
            price_data: Dict of symbol -> DataFrame
            window: Rolling window size (default: ROLLING_WINDOW)
            
        Returns:
            List of regime snapshots over time
        """
        if window is None:
            window = self.ROLLING_WINDOW
        
        # Build aligned returns
        returns_dict = {}
        for symbol, df in price_data.items():
            if df.empty or len(df) < window + 10:
                continue
            close = df['close'].astype(float)
            returns = close.pct_change().dropna()
            if len(returns) >= window:
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 10:
            return []
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(axis=0, thresh=len(returns_df.columns) * 0.7)
        returns_df = returns_df.fillna(0)
        
        regimes = []
        
        for i in range(window, len(returns_df)):
            window_data = returns_df.iloc[i-window:i]
            
            try:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(window_data)
                
                pca = PCA(n_components=self.n_components)
                pca.fit(scaled)
                
                # Get regime for this window
                var_explained = pca.explained_variance_ratio_
                
                # Simple regime detection based on variance concentration
                if var_explained[0] > 0.4:
                    regime = "Risk-Off"  # High market concentration
                elif var_explained[0] < 0.25:
                    regime = "Rotation"  # Dispersed factors
                else:
                    regime = "Risk-On"  # Normal market
                
                regimes.append({
                    'date': returns_df.index[i],
                    'regime': regime,
                    'variance_explained': list(var_explained)
                })
                
            except Exception as e:
                logger.debug(f"Rolling PCA failed at {i}: {e}")
        
        self._rolling_regimes = regimes
        return regimes
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Reduce feature dimensionality using PCA.
        
        Args:
            features: Raw features array (samples × features)
            
        Returns:
            Reduced features array (samples × n_components)
        """
        if not self._is_fitted:
            return features
        
        try:
            scaled = self.scaler.transform(features)
            return self.pca.transform(scaled)
        except Exception as e:
            logger.debug(f"Feature transform failed: {e}")
            return features
    
    def get_summary(self) -> Dict:
        """Get summary of current PCA state."""
        return {
            'is_fitted': self._is_fitted,
            'n_components': self.n_components,
            'n_stocks': len(self._symbols),
            'n_days': len(self._returns_matrix) if self._returns_matrix is not None else 0,
            'variance_explained': self.get_variance_explained(),
            'current_regime': self.get_market_regime()
        }
    
    # ==================== ADVANCED ANALYTICS ====================
    
    def find_similar_stocks(self, symbol: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find stocks with similar factor profiles using Euclidean distance.
        
        Args:
            symbol: Target stock symbol
            top_n: Number of similar stocks to return
            
        Returns:
            List of (symbol, similarity_score) tuples, higher is more similar
        """
        if not self._is_fitted or symbol not in self._symbols:
            return []
        
        idx = self._symbols.index(symbol)
        target_loadings = self._factor_loadings[idx]
        
        similarities = []
        for i, sym in enumerate(self._symbols):
            if sym == symbol:
                continue
            
            other_loadings = self._factor_loadings[i]
            # Euclidean distance, converted to similarity
            distance = np.sqrt(np.sum((target_loadings - other_loadings) ** 2))
            similarity = 1 / (1 + distance)  # 0 to 1, higher is more similar
            similarities.append((sym, round(similarity, 3)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_factor_attribution(self, symbol: str, days: int = 20) -> Dict[str, float]:
        """
        Decompose stock return into factor contributions.
        
        Args:
            symbol: Stock symbol
            days: Lookback period
            
        Returns:
            Dict with factor contributions + residual alpha
        """
        if not self._is_fitted or symbol not in self._symbols:
            return {}
        
        if self._returns_matrix is None or len(self._returns_matrix) < days:
            return {}
        
        idx = self._symbols.index(symbol)
        
        # Get stock returns and factor returns for period
        stock_returns = self._returns_matrix.iloc[-days:, self._returns_matrix.columns.get_loc(symbol)]
        total_return = stock_returns.sum()
        
        # Factor contributions
        factor_returns = self.get_factor_returns().tail(days)
        exposures = self._factor_loadings[idx]
        
        attribution = {}
        total_factor_contribution = 0
        
        for i, factor_name in enumerate(self.FACTOR_NAMES[:self.n_components]):
            if factor_name in factor_returns.columns:
                factor_ret = factor_returns[factor_name].sum()
                contribution = exposures[i] * factor_ret
                attribution[factor_name] = round(contribution * 100, 2)  # As percentage
                total_factor_contribution += contribution
        
        # Residual alpha = total return - factor explained return
        attribution['Alpha'] = round((total_return - total_factor_contribution) * 100, 2)
        attribution['Total'] = round(total_return * 100, 2)
        
        return attribution
    
    def get_regime_performance(self, symbol: str) -> Dict[str, Dict]:
        """
        Calculate historical performance by regime type.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with performance stats per regime
        """
        if not self._is_fitted or symbol not in self._symbols:
            return {}
        
        if self._returns_matrix is None or len(self._returns_matrix) < 60:
            return {}
        
        # Classify each historical period into regimes
        returns = self._returns_matrix[symbol].values
        
        # Use rolling PCA or approximate regime detection
        regime_returns = {'Risk-On': [], 'Risk-Off': [], 'Rotation': []}
        
        # Simplified: use market factor return to determine regime
        factor_returns = self.get_factor_returns()
        if factor_returns.empty or 'Market' not in factor_returns.columns:
            return {}
        
        market_returns = factor_returns['Market'].values
        
        for i in range(20, len(returns)):
            # Rolling market mean to detect regime
            rolling_market = np.mean(market_returns[max(0, i-20):i])
            stock_ret = returns[i]
            
            if rolling_market > 0.001:
                regime_returns['Risk-On'].append(stock_ret)
            elif rolling_market < -0.001:
                regime_returns['Risk-Off'].append(stock_ret)
            else:
                regime_returns['Rotation'].append(stock_ret)
        
        result = {}
        for regime, rets in regime_returns.items():
            if len(rets) > 5:
                result[regime] = {
                    'avg_return': round(np.mean(rets) * 100 * 20, 2),  # Monthly equivalent
                    'count': len(rets),
                    'win_rate': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
                }
            else:
                result[regime] = {'avg_return': 0, 'count': len(rets), 'win_rate': 50}
        
        return result
    
    def get_risk_decomposition(self, symbol: str) -> Dict[str, float]:
        """
        Decompose stock volatility by factor source.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with % of risk from each factor + idiosyncratic
        """
        if not self._is_fitted or symbol not in self._symbols:
            return {}
        
        if self._returns_matrix is None:
            return {}
        
        idx = self._symbols.index(symbol)
        exposures = self._factor_loadings[idx]
        
        # Get factor variances
        factor_returns = self.get_factor_returns()
        if factor_returns.empty:
            return {}
        
        factor_vars = factor_returns.var().values[:self.n_components]
        
        # Stock variance
        stock_var = self._returns_matrix[symbol].var()
        if stock_var == 0:
            return {}
        
        # Factor contribution to variance: beta^2 * factor_variance
        decomposition = {}
        total_factor_var = 0
        
        for i, factor_name in enumerate(self.FACTOR_NAMES[:self.n_components]):
            factor_contribution = (exposures[i] ** 2) * factor_vars[i]
            pct = min(100, max(0, factor_contribution / stock_var * 100))
            decomposition[factor_name] = round(pct, 1)
            total_factor_var += factor_contribution
        
        # Idiosyncratic risk
        idio_pct = max(0, 100 - sum(decomposition.values()))
        decomposition['Idiosyncratic'] = round(idio_pct, 1)
        
        return decomposition
    
    def calculate_what_if(self, symbol: str, factor: str, change_pct: float) -> float:
        """
        Simulate stock return if a factor moves by given percentage.
        
        Args:
            symbol: Stock symbol
            factor: Factor name (Market, Size, etc.)
            change_pct: Factor change in percentage (e.g., 5 for +5%)
            
        Returns:
            Expected stock return in percentage
        """
        if not self._is_fitted or symbol not in self._symbols:
            return 0.0
        
        exposures = self.get_factor_exposures(symbol)
        exposure = exposures.get(factor, 0)
        
        # Stock return ≈ exposure × factor return
        expected_return = exposure * (change_pct / 100) * 100
        return round(expected_return, 2)
    
    def get_market_average_exposures(self) -> Dict[str, float]:
        """Get average factor exposures across all stocks."""
        if not self._is_fitted:
            return {}
        
        avg_exposures = np.mean(self._factor_loadings, axis=0)
        return {
            self.FACTOR_NAMES[i]: round(float(avg_exposures[i]), 3)
            for i in range(self.n_components)
        }
