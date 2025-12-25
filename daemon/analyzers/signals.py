# Signals Aggregator for Daemon

import logging
from typing import Dict, List
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)


async def get_top_signals(config: Config, limit: int = 5) -> str:
    """Get top bullish and bearish signals."""
    from analyzers.pathway import PathwayAnalyzer
    
    analyzer = PathwayAnalyzer(config)
    
    results = []
    
    for symbol in config.default_watchlist:
        try:
            result = await analyzer.synthesize(symbol)
            if 'error' not in result:
                ret = result['predictions']['2d']['expected_return']
                bull_prob = result['predictions']['2d']['bull']['probability']
                results.append((symbol, ret, bull_prob))
        except Exception as e:
            logger.error(f"Signal error for {symbol}: {e}")
    
    # Sort by expected return
    results.sort(key=lambda x: x[1], reverse=True)
    
    top_bull = results[:limit]
    top_bear = results[-limit:][::-1]
    
    lines = [
        "🎯 <b>TOP SIGNALS</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        "<b>🟢 Most Bullish:</b>"
    ]
    
    for sym, ret, prob in top_bull:
        if ret > 0:
            lines.append(f"• {sym}: +{ret:.1f}% ({prob:.0f}% Bull)")
    
    lines.extend(["", "<b>🔴 Most Bearish:</b>"])
    
    for sym, ret, prob in top_bear:
        if ret < 0:
            lines.append(f"• {sym}: {ret:.1f}% ({100-prob:.0f}% Bear)")
    
    return "\n".join(lines)


async def generate_evening_digest(config: Config) -> str:
    """Generate evening digest with tomorrow's watchlist."""
    from analyzers.pathway import PathwayAnalyzer
    
    analyzer = PathwayAnalyzer(config)
    now = datetime.now()
    
    # Get top opportunities
    opportunities = []
    
    for symbol in config.default_watchlist:
        try:
            result = await analyzer.synthesize(symbol)
            if 'error' not in result:
                ret = result['predictions']['2d']['expected_return']
                bull_prob = result['predictions']['2d']['bull']['probability']
                
                # Score = return * probability
                score = ret * (bull_prob / 100)
                opportunities.append((symbol, ret, bull_prob, score))
        except:
            pass
    
    # Sort by score
    opportunities.sort(key=lambda x: abs(x[3]), reverse=True)
    top = opportunities[:5]
    
    lines = [
        "🌙 <b>EVENING DIGEST</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"<i>{now.strftime('%Y-%m-%d')}</i>",
        "",
        "<b>📋 Tomorrow's Watch:</b>"
    ]
    
    for sym, ret, prob, score in top:
        emoji = '🟢' if ret > 0 else '🔴'
        sign = '+' if ret > 0 else ''
        lines.append(f"{emoji} <b>{sym}</b>: {sign}{ret:.1f}% (2D) | {prob:.0f}% Bull")
    
    lines.extend([
        "",
        "<b>⏰ Schedule:</b>",
        "• 09:30 - Pre-market brief",
        "• 10:00 - Market open",
        "• 12:00 - Midday synthesis",
        "",
        "Good night! 🌃"
    ])
    
    return "\n".join(lines)
