# Market Intelligence Analyzer for Daemon

import logging
from typing import Dict, List
from datetime import datetime

from config import Config

logger = logging.getLogger(__name__)


async def generate_pre_market_brief(config: Config) -> str:
    """Generate pre-market briefing."""
    now = datetime.now()
    
    return f"""
📰 <b>PRE-MARKET BRIEF</b>
━━━━━━━━━━━━━━━━━━━━━━
{now.strftime('%Y-%m-%d')}

<b>Watchlist Summary:</b>
{', '.join(config.default_watchlist[:10])}

<b>Key Levels to Watch:</b>
• Market opens in 30 minutes
• Use /pathway SYMBOL for predictions

<b>Sector Focus:</b>
Banking | Oil & Gas | Consumer Goods
    """.strip()


async def generate_market_summary(config: Config) -> str:
    """Generate on-demand market summary."""
    from analyzers.pathway import PathwayAnalyzer
    
    analyzer = PathwayAnalyzer(config)
    
    bullish = []
    bearish = []
    
    for symbol in config.default_watchlist[:10]:
        try:
            result = await analyzer.synthesize(symbol)
            if 'error' not in result:
                ret = result['predictions']['2d']['expected_return']
                if ret > 1:
                    bullish.append(f"{symbol} (+{ret:.1f}%)")
                elif ret < -1:
                    bearish.append(f"{symbol} ({ret:.1f}%)")
        except:
            pass
    
    return f"""
📊 <b>MARKET SYNTHESIS</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>🟢 Bullish Signals:</b>
{chr(10).join(bullish) if bullish else 'None detected'}

<b>🔴 Bearish Signals:</b>
{chr(10).join(bearish) if bearish else 'None detected'}

Use /pathway SYMBOL for detailed analysis
    """.strip()


async def generate_eod_summary(config: Config) -> str:
    """Generate end-of-day summary."""
    now = datetime.now()
    
    return f"""
📊 <b>END OF DAY SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━━━
{now.strftime('%Y-%m-%d')}

<b>Session Stats:</b>
• Total watchlist: {len(config.default_watchlist)} symbols
• Alerts sent: Check logs

<b>Tomorrow's Setup:</b>
Evening digest at 16:00 WAT with predictions.

Use /summary for detailed analysis.
    """.strip()
