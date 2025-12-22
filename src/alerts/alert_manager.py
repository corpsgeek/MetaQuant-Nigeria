"""
Alert Manager for MetaQuant Nigeria.
Handles push notifications for market events.
"""

import os
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage market alerts and notifications."""
    
    # Alert cooldown in minutes (don't repeat same alert)
    COOLDOWN_MINUTES = 30
    
    # Alert thresholds
    VOLUME_SPIKE_THRESHOLD = 5.0  # 5x RVOL
    MAJOR_MOVE_THRESHOLD = 5.0    # ±5% price change
    HEALTH_SHIFT_THRESHOLD = 20   # 20 point shift in health score
    
    def __init__(self):
        self.last_alerts = {}  # Track recent alerts to prevent spam
        self.last_health_score = None
        self.last_rotation_phase = None
        self.alert_history = []  # Store alert history
    
    def check_all_alerts(
        self,
        snapshot: Dict[str, Any],
        stocks_list: List[Dict],
        sector_rankings: List[Dict] = None,
        rotation_phase: str = None
    ) -> List[Dict]:
        """
        Check all alert conditions and send notifications.
        
        Args:
            snapshot: Market snapshot data
            stocks_list: List of all stocks
            sector_rankings: Optional sector data
            rotation_phase: Current rotation phase
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check volume spikes
        volume_alerts = self._check_volume_spikes(stocks_list)
        triggered_alerts.extend(volume_alerts)
        
        # Check major price moves
        price_alerts = self._check_major_moves(stocks_list)
        triggered_alerts.extend(price_alerts)
        
        # Check market health shift
        health_alert = self._check_health_shift(snapshot)
        if health_alert:
            triggered_alerts.append(health_alert)
        
        # Check sector rotation phase change
        if rotation_phase:
            phase_alert = self._check_rotation_phase(rotation_phase)
            if phase_alert:
                triggered_alerts.append(phase_alert)
        
        # Send notifications for all triggered alerts
        for alert in triggered_alerts:
            self._send_notification(alert)
            self.alert_history.append({
                **alert,
                'timestamp': datetime.now().isoformat()
            })
        
        return triggered_alerts
    
    def _check_volume_spikes(self, stocks_list: List[Dict]) -> List[Dict]:
        """Check for volume spike alerts."""
        alerts = []
        
        for stock in stocks_list:
            symbol = stock.get('symbol', '?')
            rvol = stock.get('rvol', 0) or 0
            
            if rvol >= self.VOLUME_SPIKE_THRESHOLD:
                alert_key = f"volume_{symbol}"
                
                if self._can_alert(alert_key):
                    chg = stock.get('change', 0) or 0
                    direction = "buying" if chg > 0 else "selling"
                    
                    alerts.append({
                        'type': 'volume_spike',
                        'title': f"🔥 Volume Spike: {symbol}",
                        'message': f"{symbol} has {rvol:.1f}x normal volume with heavy {direction}",
                        'symbol': symbol,
                        'rvol': rvol,
                        'priority': 'high' if rvol >= 10 else 'medium'
                    })
                    self._mark_alerted(alert_key)
        
        return alerts
    
    def _check_major_moves(self, stocks_list: List[Dict]) -> List[Dict]:
        """Check for major price moves."""
        alerts = []
        
        for stock in stocks_list:
            symbol = stock.get('symbol', '?')
            change = stock.get('change', 0) or 0
            
            if abs(change) >= self.MAJOR_MOVE_THRESHOLD:
                alert_key = f"move_{symbol}"
                
                if self._can_alert(alert_key):
                    direction = "📈 Surging" if change > 0 else "📉 Plunging"
                    
                    alerts.append({
                        'type': 'major_move',
                        'title': f"{direction}: {symbol}",
                        'message': f"{symbol} moved {change:+.1f}% today",
                        'symbol': symbol,
                        'change': change,
                        'priority': 'high' if abs(change) >= 10 else 'medium'
                    })
                    self._mark_alerted(alert_key)
        
        return alerts
    
    def _check_health_shift(self, snapshot: Dict) -> Optional[Dict]:
        """Check for significant market health shift."""
        gainers = snapshot.get('gainers', 0)
        losers = snapshot.get('losers', 0)
        total = snapshot.get('total_stocks', 1)
        
        current_health = ((gainers - losers) / total * 50 + 50) if total > 0 else 50
        
        if self.last_health_score is not None:
            shift = current_health - self.last_health_score
            
            if abs(shift) >= self.HEALTH_SHIFT_THRESHOLD:
                alert_key = "health_shift"
                
                if self._can_alert(alert_key):
                    if shift > 0:
                        title = "🟢 Market Turning Bullish"
                        msg = f"Market health jumped from {self.last_health_score:.0f} to {current_health:.0f}"
                    else:
                        title = "🔴 Market Turning Bearish"
                        msg = f"Market health dropped from {self.last_health_score:.0f} to {current_health:.0f}"
                    
                    self._mark_alerted(alert_key)
                    self.last_health_score = current_health
                    
                    return {
                        'type': 'health_shift',
                        'title': title,
                        'message': msg,
                        'from_score': self.last_health_score,
                        'to_score': current_health,
                        'priority': 'high'
                    }
        
        self.last_health_score = current_health
        return None
    
    def _check_rotation_phase(self, current_phase: str) -> Optional[Dict]:
        """Check for sector rotation phase change."""
        if self.last_rotation_phase is not None and self.last_rotation_phase != current_phase:
            alert_key = "rotation_phase"
            
            if self._can_alert(alert_key):
                phase_descriptions = {
                    'EARLY': '💹 Recovery Phase (Risk-On)',
                    'MID': '📈 Expansion Phase (Growth)',
                    'LATE': '⚠️ Late Cycle (Defensive)',
                    'CONTRACTION': '🛡️ Contraction (Safety)'
                }
                
                desc = phase_descriptions.get(current_phase, current_phase)
                
                self._mark_alerted(alert_key)
                self.last_rotation_phase = current_phase
                
                return {
                    'type': 'rotation_phase',
                    'title': f"🔄 Sector Rotation: {current_phase}",
                    'message': f"Market entering {desc}",
                    'from_phase': self.last_rotation_phase,
                    'to_phase': current_phase,
                    'priority': 'high'
                }
        
        self.last_rotation_phase = current_phase
        return None
    
    def _can_alert(self, alert_key: str) -> bool:
        """Check if we can send this alert (not in cooldown)."""
        if alert_key not in self.last_alerts:
            return True
        
        last_time = self.last_alerts[alert_key]
        now = datetime.now()
        
        return (now - last_time) > timedelta(minutes=self.COOLDOWN_MINUTES)
    
    def _mark_alerted(self, alert_key: str):
        """Mark an alert as sent."""
        self.last_alerts[alert_key] = datetime.now()
    
    def _send_notification(self, alert: Dict):
        """Send system notification (macOS)."""
        title = alert.get('title', 'MetaQuant Alert')
        message = alert.get('message', '')
        
        try:
            # macOS native notification using osascript
            script = f'''
            display notification "{message}" with title "{title}" sound name "Glass"
            '''
            subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                timeout=5
            )
            logger.info(f"Notification sent: {title}")
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def clear_cooldowns(self):
        """Clear all alert cooldowns (useful for testing)."""
        self.last_alerts.clear()
    
    def test_notification(self):
        """Send a test notification."""
        self._send_notification({
            'title': '🧪 MetaQuant Test',
            'message': 'Notifications are working!'
        })
