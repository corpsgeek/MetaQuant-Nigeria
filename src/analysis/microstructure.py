"""
Microstructure Analysis Module for MetaQuant Nigeria.

Provides calculations for market microstructure metrics like:
- Relative Volume (RVOL)
- Price Momentum & Rate of Change
- Breadth Indicators
- Volume Leaders Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_relative_volume(
    current_volume: float,
    avg_volume: float,
    lookback_days: int = 20
) -> float:
    """
    Calculate Relative Volume (RVOL).
    
    RVOL = Current Volume / Average Volume
    
    Args:
        current_volume: Today's volume
        avg_volume: Average volume over lookback period
        lookback_days: Period for average (default 20 days)
        
    Returns:
        RVOL multiplier (e.g., 2.5 means 2.5x average volume)
    """
    if avg_volume is None or avg_volume <= 0:
        return 0.0
    return current_volume / avg_volume


def calculate_rvol_from_history(
    stock_id: int,
    current_volume: float,
    db,
    lookback_days: int = 20
) -> float:
    """
    Calculate RVOL using historical data from database.
    
    Args:
        stock_id: Database stock ID
        current_volume: Today's volume
        db: DatabaseManager instance
        lookback_days: Period for average
        
    Returns:
        RVOL multiplier
    """
    try:
        # Get historical volume data
        history = db.get_price_history(stock_id, days=lookback_days)
        
        if not history or len(history) < 5:
            return 0.0
        
        # Calculate average volume
        volumes = [h.get('volume', 0) or 0 for h in history]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        return calculate_relative_volume(current_volume, avg_volume)
        
    except Exception as e:
        logger.error(f"Error calculating RVOL for stock {stock_id}: {e}")
        return 0.0


def calculate_momentum(
    prices: List[float],
    period: int = 5
) -> Dict[str, float]:
    """
    Calculate price momentum indicators.
    
    Args:
        prices: List of prices (most recent first)
        period: Lookback period for momentum
        
    Returns:
        Dictionary with momentum metrics
    """
    if not prices or len(prices) < period + 1:
        return {
            'roc': 0.0,
            'velocity': 0.0,
            'trend': 'neutral'
        }
    
    current_price = prices[0]
    past_price = prices[period] if len(prices) > period else prices[-1]
    
    # Rate of Change (ROC)
    roc = ((current_price - past_price) / past_price * 100) if past_price > 0 else 0
    
    # Velocity (smoothed price change)
    recent_changes = []
    for i in range(min(period, len(prices) - 1)):
        if prices[i + 1] > 0:
            change = (prices[i] - prices[i + 1]) / prices[i + 1] * 100
            recent_changes.append(change)
    
    velocity = sum(recent_changes) / len(recent_changes) if recent_changes else 0
    
    # Trend classification
    if roc > 5:
        trend = 'strong_up'
    elif roc > 1:
        trend = 'up'
    elif roc < -5:
        trend = 'strong_down'
    elif roc < -1:
        trend = 'down'
    else:
        trend = 'neutral'
    
    return {
        'roc': round(roc, 2),
        'velocity': round(velocity, 2),
        'trend': trend
    }


def get_momentum_from_history(
    stock_id: int,
    current_price: float,
    db,
    period: int = 5
) -> Dict[str, float]:
    """
    Calculate momentum using historical data.
    
    Args:
        stock_id: Database stock ID
        current_price: Current price
        db: DatabaseManager instance
        period: Lookback period
        
    Returns:
        Momentum metrics dictionary
    """
    try:
        history = db.get_price_history(stock_id, days=period + 5)
        
        if not history:
            return {'roc': 0.0, 'velocity': 0.0, 'trend': 'neutral'}
        
        # Build price list (most recent first)
        prices = [current_price] + [h.get('close', 0) or 0 for h in history]
        
        return calculate_momentum(prices, period)
        
    except Exception as e:
        logger.error(f"Error calculating momentum for stock {stock_id}: {e}")
        return {'roc': 0.0, 'velocity': 0.0, 'trend': 'neutral'}


def calculate_breadth_indicators(
    stocks_data: List[Dict[str, Any]],
    db=None
) -> Dict[str, Any]:
    """
    Calculate market breadth indicators.
    
    Args:
        stocks_data: List of stock dictionaries with change data
        db: Optional DatabaseManager for historical lookups
        
    Returns:
        Dictionary with breadth metrics
    """
    if not stocks_data:
        return {
            'advances': 0,
            'declines': 0,
            'unchanged': 0,
            'ad_ratio': 0.0,
            'ad_line': 0,
            'pct_above_sma': 0.0,
            'new_highs': 0,
            'new_lows': 0,
        }
    
    advances = sum(1 for s in stocks_data if (s.get('change') or 0) > 0)
    declines = sum(1 for s in stocks_data if (s.get('change') or 0) < 0)
    unchanged = len(stocks_data) - advances - declines
    
    # A/D Ratio
    ad_ratio = advances / declines if declines > 0 else advances
    
    # A/D Line (cumulative)
    ad_line = advances - declines
    
    # Calculate % above 20-day SMA (simplified - use current price vs avg)
    above_sma = 0
    for stock in stocks_data:
        # If we have SMA data from tradingview-ta, use it
        sma_20 = stock.get('sma_20') or stock.get('SMA20')
        close = stock.get('close', 0)
        if sma_20 and close and close > sma_20:
            above_sma += 1
    
    pct_above_sma = (above_sma / len(stocks_data) * 100) if stocks_data else 0
    
    # New highs/lows (would need 52-week data, using placeholder)
    # In a real implementation, we'd compare to 52-week high/low from database
    new_highs = 0
    new_lows = 0
    
    return {
        'advances': advances,
        'declines': declines,
        'unchanged': unchanged,
        'ad_ratio': round(ad_ratio, 2),
        'ad_line': ad_line,
        'pct_above_sma': round(pct_above_sma, 1),
        'new_highs': new_highs,
        'new_lows': new_lows,
    }


def identify_volume_leaders(
    stocks_data: List[Dict[str, Any]],
    db=None,
    rvol_threshold: float = 2.0,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Identify stocks with unusual volume activity.
    
    Args:
        stocks_data: List of stock dictionaries
        db: DatabaseManager for historical lookups
        rvol_threshold: Minimum RVOL to be considered a leader
        top_n: Number of leaders to return
        
    Returns:
        List of volume leader stocks with RVOL
    """
    volume_leaders = []
    
    for stock in stocks_data:
        symbol = stock.get('symbol', '')
        current_volume = stock.get('volume', 0) or 0
        
        # Skip if no volume
        if current_volume <= 0:
            continue
        
        # Try to get RVOL from database if available
        rvol = 0.0
        if db:
            try:
                stock_record = db.get_stock(symbol)
                if stock_record:
                    rvol = calculate_rvol_from_history(
                        stock_record['id'],
                        current_volume,
                        db
                    )
            except:
                pass
        
        # If no RVOL calculated, estimate from relative position
        if rvol == 0:
            # Use a simple heuristic based on volume ranking
            rvol = 1.0
        
        if rvol >= rvol_threshold or current_volume > 10_000_000:
            volume_leaders.append({
                'symbol': symbol,
                'name': stock.get('name', symbol),
                'close': stock.get('close', 0),
                'change': stock.get('change', 0),
                'volume': current_volume,
                'rvol': round(rvol, 1) if rvol > 0 else None
            })
    
    # Sort by volume descending
    volume_leaders.sort(key=lambda x: x['volume'], reverse=True)
    
    return volume_leaders[:top_n]


def calculate_sector_performance(
    stocks_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate sector-level performance metrics.
    
    Args:
        stocks_data: List of stock dictionaries with sector data
        
    Returns:
        Dictionary of sector -> performance metrics
    """
    sectors = {}
    
    for stock in stocks_data:
        sector = stock.get('sector', 'Unknown') or 'Unknown'
        if sector not in sectors:
            sectors[sector] = {
                'stocks': [],
                'total_change': 0.0,
                'gainers': 0,
                'losers': 0,
                'total_volume': 0,
            }
        
        change = stock.get('change', 0) or 0
        volume = stock.get('volume', 0) or 0
        
        sectors[sector]['stocks'].append(stock)
        sectors[sector]['total_change'] += change
        sectors[sector]['total_volume'] += volume
        
        if change > 0:
            sectors[sector]['gainers'] += 1
        elif change < 0:
            sectors[sector]['losers'] += 1
    
    # Calculate averages
    result = {}
    for sector, data in sectors.items():
        num_stocks = len(data['stocks'])
        result[sector] = {
            'avg_change': round(data['total_change'] / num_stocks, 2) if num_stocks > 0 else 0,
            'num_stocks': num_stocks,
            'gainers': data['gainers'],
            'losers': data['losers'],
            'total_volume': data['total_volume'],
        }
    
    return result
