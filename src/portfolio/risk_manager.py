"""
Risk Manager for MetaQuant Nigeria Portfolio.
Handles position sizing, drawdown monitoring, and risk limits.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_drawdown_pct: float = 0.10  # 10%
    max_position_pct: float = 0.15  # 15% max per position
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    max_correlation: float = 0.70  # Don't overload correlated assets
    max_sector_exposure: float = 0.30  # 30% max per sector


class RiskManager:
    """
    Portfolio risk management.
    
    Features:
    - Position sizing based on volatility and conviction
    - Drawdown monitoring and circuit breaker
    - Correlation-aware allocation
    - Daily VaR estimation
    """
    
    def __init__(self, capital: float, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.
        
        Args:
            capital: Total portfolio capital
            limits: Risk limits configuration
        """
        self.capital = capital
        self.limits = limits or RiskLimits()
        
        # Tracking
        self.peak_equity = capital
        self.current_equity = capital
        self.current_drawdown = 0.0
        self.drawdown_triggered = False
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.sector_exposure: Dict[str, float] = {}
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float,
        conviction: float = 1.0,
        volatility: float = 0.02
    ) -> int:
        """
        Calculate position size based on risk limits.
        
        Uses Kelly-inspired sizing:
        - Base size from risk per trade
        - Adjusted by conviction (signal strength)
        - Capped by max position size
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss_pct: Stop loss as percentage (e.g., 0.05 for 5%)
            conviction: Signal strength 0-1
            volatility: Stock volatility (daily std)
            
        Returns:
            Number of shares to buy
        """
        if self.drawdown_triggered:
            logger.warning("Drawdown circuit breaker active - no new positions")
            return 0
        
        if entry_price <= 0 or stop_loss_pct <= 0:
            return 0
        
        # Risk amount = capital * risk per trade * conviction
        risk_amount = self.capital * self.limits.risk_per_trade_pct * conviction
        
        # Position size from risk amount and stop loss
        # If stop is 5%, we can buy (risk_amount / 0.05) worth
        position_value = risk_amount / stop_loss_pct
        
        # Cap at max position size
        max_position = self.capital * self.limits.max_position_pct
        position_value = min(position_value, max_position)
        
        # Adjust for volatility - reduce size for volatile stocks
        if volatility > 0.03:  # High vol
            position_value *= 0.7
        elif volatility > 0.02:
            position_value *= 0.85
        
        # Calculate shares
        shares = int(position_value / entry_price)
        
        return max(0, shares)
    
    def update_equity(self, new_equity: float) -> Dict[str, Any]:
        """
        Update current equity and check drawdown limits.
        
        Args:
            new_equity: Current portfolio value
            
        Returns:
            Dict with drawdown info and any alerts
        """
        self.current_equity = new_equity
        
        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        # Calculate drawdown
        self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity
        
        # Check circuit breaker
        alerts = []
        if self.current_drawdown >= self.limits.max_drawdown_pct:
            if not self.drawdown_triggered:
                self.drawdown_triggered = True
                alerts.append({
                    'type': 'CIRCUIT_BREAKER',
                    'message': f'Max drawdown {self.limits.max_drawdown_pct:.0%} exceeded',
                    'drawdown': self.current_drawdown
                })
        else:
            # Reset if recovered
            if self.current_drawdown < self.limits.max_drawdown_pct * 0.5:
                self.drawdown_triggered = False
        
        return {
            'equity': new_equity,
            'peak': self.peak_equity,
            'drawdown': self.current_drawdown,
            'drawdown_pct': self.current_drawdown * 100,
            'circuit_breaker': self.drawdown_triggered,
            'alerts': alerts
        }
    
    def check_position_allowed(
        self, 
        symbol: str, 
        sector: str,
        position_value: float
    ) -> tuple[bool, str]:
        """
        Check if a new position is allowed within risk limits.
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            position_value: Proposed position value
            
        Returns:
            Tuple of (allowed, reason)
        """
        if self.drawdown_triggered:
            return False, "Drawdown circuit breaker active"
        
        # Check position size limit
        if position_value > self.capital * self.limits.max_position_pct:
            return False, f"Position exceeds {self.limits.max_position_pct:.0%} limit"
        
        # Check sector exposure
        current_sector = self.sector_exposure.get(sector, 0)
        if current_sector + position_value > self.capital * self.limits.max_sector_exposure:
            return False, f"Sector exposure would exceed {self.limits.max_sector_exposure:.0%}"
        
        return True, "Allowed"
    
    def add_position(self, symbol: str, sector: str, value: float):
        """Track a new position."""
        self.positions[symbol] = {'value': value, 'sector': sector}
        self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + value
    
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            sector = pos.get('sector', 'Unknown')
            value = pos.get('value', 0)
            self.sector_exposure[sector] = max(0, self.sector_exposure.get(sector, 0) - value)
            del self.positions[symbol]
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        return {
            'capital': self.capital,
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': round(self.current_drawdown * 100, 2),
            'circuit_breaker_active': self.drawdown_triggered,
            'positions': len(self.positions),
            'sector_exposure': dict(self.sector_exposure),
            'limits': {
                'max_drawdown': self.limits.max_drawdown_pct * 100,
                'max_position': self.limits.max_position_pct * 100,
                'risk_per_trade': self.limits.risk_per_trade_pct * 100
            }
        }
    
    def estimate_var(
        self, 
        returns: pd.Series, 
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Estimate Value at Risk.
        
        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95)
            horizon_days: Time horizon in days
            
        Returns:
            VaR as positive number (potential loss)
        """
        if returns.empty:
            return 0
        
        # Historical VaR
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        var_scaled = var_pct * np.sqrt(horizon_days)
        
        return abs(var_scaled * self.current_equity)
