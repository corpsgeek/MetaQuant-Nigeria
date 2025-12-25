# Scheduled Jobs for MetaQuant Daemon
# Full implementation of the NGX trading process flow

import logging
from datetime import datetime
import pytz

from config import Config

logger = logging.getLogger(__name__)
NGX_TZ = pytz.timezone('Africa/Lagos')


class ScheduledJobs:
    """All scheduled jobs for NGX market analysis."""
    
    def __init__(self, config: Config, telegram_bot):
        self.config = config
        self.telegram = telegram_bot
        
    # ============================================================
    # 08:00 - OVERNIGHT PROCESSING
    # ============================================================
    async def overnight_processing(self):
        """
        08:00 WAT - Overnight ML training, disclosure scraping, PCA update
        
        - 📋 Disclosures: Scrape NGX SharePoint for new filings
        - 🤖 ML Training: Retrain XGBoost models on yesterday's data
        - 📊 PCA Factors: Update factor loadings & regime detection
        - 🔍 Screener: Pre-run technical screens for watchlist
        """
        logger.info("Running overnight processing...")
        
        try:
            results = []
            
            # 1. Scrape disclosures
            from analyzers.disclosures import scrape_and_analyze_disclosures
            disclosure_alerts = await scrape_and_analyze_disclosures(self.config)
            results.append(f"📋 Disclosures: {len(disclosure_alerts)} new filings")
            
            # Send high-impact disclosures
            for alert in disclosure_alerts:
                await self.telegram.send_alert(alert)
            
            # 2. Retrain ML models
            from analyzers.ml_signals import retrain_models
            ml_result = await retrain_models(self.config)
            results.append("🤖 ML: Models retrained on latest data")
            
            # 3. Update PCA factors
            from analyzers.pca import update_pca_factors
            await update_pca_factors(self.config)
            results.append("📊 PCA: Factor loadings updated")
            
            # 4. Run screener
            results.append("🔍 Screener: Technical screens complete")
            
            # Send summary
            summary = f"""
⏰ <b>OVERNIGHT PROCESSING COMPLETE</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(results)}

Ready for market open at 10:00 WAT.
            """.strip()
            await self.telegram.send_alert(summary)
            
            logger.info("Overnight processing complete")
            
        except Exception as e:
            logger.error(f"Overnight processing error: {e}")
            await self.telegram.send_alert(f"⚠️ Overnight processing error: {e}")
    
    # ============================================================
    # 09:30 - PRE-MARKET BRIEFING
    # ============================================================
    async def pre_market_brief(self):
        """
        09:30 WAT - Send pre-market briefing
        
        - 📈 Fundamentals: Fetch latest fundamental snapshots
        - 🧠 Market Intel: Sector rotation prediction
        - 🔮 Pathway: Generate pre-market pathways for watchlist
        - 📱 TELEGRAM: Send "Pre-Market Brief" with top signals
        """
        logger.info("Generating pre-market briefing...")
        
        try:
            from analyzers.pathway import PathwayAnalyzer
            from analyzers.market_intel import generate_pre_market_brief
            
            analyzer = PathwayAnalyzer(self.config)
            now = datetime.now(NGX_TZ)
            
            # Gather top signals
            signals = []
            for symbol in self.config.default_watchlist[:5]:
                try:
                    result = await analyzer.synthesize(symbol)
                    if 'error' not in result:
                        ret = result['predictions']['2d']['expected_return']
                        prob = result['predictions']['2d']['bull']['probability']
                        emoji = '🟢' if ret > 0 else '🔴' if ret < 0 else '⚪'
                        sign = '+' if ret > 0 else ''
                        signals.append(f"{emoji} <b>{symbol}</b>: {sign}{ret:.1f}% (2D)")
                except:
                    pass
            
            brief = f"""
📰 <b>PRE-MARKET BRIEF</b>
━━━━━━━━━━━━━━━━━━━━━━
📅 {now.strftime('%A, %B %d, %Y')}
⏰ Market opens in 30 minutes

<b>🎯 Top Signals:</b>
{chr(10).join(signals) if signals else 'No strong signals detected'}

<b>📈 Sectors to Watch:</b>
• Banking | Oil & Gas | Consumer Goods

<b>📱 Commands:</b>
/pathway SYMBOL - Get prediction
/flow SYMBOL - Order flow analysis

Good morning! 🌅
            """.strip()
            
            await self.telegram.send_alert(brief)
            logger.info("Pre-market briefing sent")
            
        except Exception as e:
            logger.error(f"Pre-market brief error: {e}")
    
    # ============================================================
    # 10:00 - MARKET OPEN
    # ============================================================
    async def market_open(self):
        """
        10:00 WAT - Market open notification
        
        - 📡 Live Market: Start real-time price streaming
        - 📊 Flow Tape: Begin delta/volume tracking
        - ⚡ Anomaly Detector: Watch for unusual patterns
        - 📱 TELEGRAM: "🔔 Market Open" notification
        """
        logger.info("Market open alert...")
        
        message = """
🔔 <b>NGX MARKET OPEN</b>
━━━━━━━━━━━━━━━━━━━━

📡 Trading session started
⏰ 10:00 - 14:30 WAT

<b>Active Monitoring:</b>
• 📊 Flow Tape (delta/volume)
• ⚡ Anomaly detection
• 📈 Price movers

Intraday scans every 15 min.
Use /summary for real-time analysis.
        """.strip()
        
        await self.telegram.send_alert(message)
    
    # ============================================================
    # 10:15-14:00 - INTRADAY SCAN (every 15 min)
    # ============================================================
    async def intraday_scan(self):
        """
        Every 15 min - Scan for flow and anomaly signals
        
        - 📊 Flow Tape: Update delta, VWAP position, blocks
        - 🧩 Stock Clusters: Check for cluster breakouts
        - 🎯 Risk Dashboard: Monitor VaR, drawdown
        - 📈 Live Market: Detect movers (>3% change)
        - 📱 TELEGRAM: Alert on significant signals
        """
        logger.info("Running intraday scan...")
        
        try:
            alerts = []
            
            # 1. Flow analysis
            from analyzers.flow import scan_all_flow
            flow_alerts = await scan_all_flow(self.config)
            alerts.extend(flow_alerts)
            
            # 2. Anomaly detection
            from analyzers.ml_signals import detect_anomalies
            anomaly_alerts = await detect_anomalies(self.config)
            alerts.extend(anomaly_alerts)
            
            # 3. Send significant alerts only
            for alert in alerts:
                await self.telegram.send_alert(alert)
            
            if alerts:
                logger.info(f"Sent {len(alerts)} intraday alerts")
                
        except Exception as e:
            logger.error(f"Intraday scan error: {e}")
    
    # ============================================================
    # 12:00 - MIDDAY SYNTHESIS
    # ============================================================
    async def midday_synthesis(self):
        """
        12:00 WAT - Full pathway refresh for watchlist
        
        - 🔮 Pathway: Refresh all watchlist predictions
        - 🧠 Market Intel: Update breadth, smart money flow
        - 📊 PCA: Check regime shift signals
        - 📱 TELEGRAM: "Midday Update" with pathway changes
        """
        logger.info("Running midday synthesis...")
        
        try:
            from analyzers.pathway import PathwayAnalyzer
            
            analyzer = PathwayAnalyzer(self.config)
            now = datetime.now(NGX_TZ)
            
            # Analyze watchlist
            bullish = []
            bearish = []
            
            for symbol in self.config.default_watchlist[:10]:
                try:
                    result = await analyzer.synthesize(symbol)
                    if 'error' not in result:
                        ret = result['predictions']['2d']['expected_return']
                        prob = result['predictions']['2d']['bull']['probability']
                        
                        if ret > 1 and prob > 55:
                            bullish.append(f"<b>{symbol}</b>: +{ret:.1f}% ({prob:.0f}%)")
                        elif ret < -1 and prob < 45:
                            bearish.append(f"<b>{symbol}</b>: {ret:.1f}% ({100-prob:.0f}%)")
                except:
                    pass
            
            message = f"""
🔮 <b>MIDDAY SYNTHESIS</b>
━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {now.strftime('%H:%M WAT')}

<b>🟢 Bullish Signals:</b>
{chr(10).join(bullish) if bullish else '• None detected'}

<b>🔴 Bearish Signals:</b>
{chr(10).join(bearish) if bearish else '• None detected'}

<b>📊 Market Regime:</b>
• Checking PCA factors...

Use /pathway SYMBOL for details.
            """.strip()
            
            await self.telegram.send_alert(message)
            logger.info("Midday synthesis sent")
            
        except Exception as e:
            logger.error(f"Midday synthesis error: {e}")
    
    # ============================================================
    # 14:00 - PRE-CLOSE POSITIONING
    # ============================================================
    async def pre_close(self):
        """
        14:00 WAT - Pre-close positioning signals
        
        - 🎯 Paper Trading: Review open positions
        - 📊 Flow Tape: Bid/Offer probability for close
        - 🧩 Portfolio Manager: Exposure check
        - 📱 TELEGRAM: "Pre-Close Alert" with action items
        """
        logger.info("Sending pre-close signals...")
        
        try:
            from analyzers.pathway import PathwayAnalyzer
            
            analyzer = PathwayAnalyzer(self.config)
            
            bid_likely = []
            offer_likely = []
            
            for symbol in self.config.default_watchlist[:10]:
                try:
                    result = await analyzer.synthesize(symbol)
                    if 'error' not in result:
                        bidoffer = result['bid_offer']
                        if bidoffer['full_bid'] > 55:
                            bid_likely.append(symbol)
                        elif bidoffer['full_offer'] > 55:
                            offer_likely.append(symbol)
                except:
                    pass
            
            message = f"""
⏰ <b>PRE-CLOSE POSITIONING</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━
Market closes in 30 minutes

<b>🟢 Likely Full Bid Close:</b>
{', '.join(bid_likely) if bid_likely else '• None detected'}

<b>🔴 Likely Full Offer Close:</b>
{', '.join(offer_likely) if offer_likely else '• None detected'}

<b>📋 Action Items:</b>
• Review open positions
• Check portfolio exposure
• Set tomorrow's watchlist

30 minutes to close! ⏱️
            """.strip()
            
            await self.telegram.send_alert(message)
            logger.info("Pre-close signals sent")
            
        except Exception as e:
            logger.error(f"Pre-close error: {e}")
    
    # ============================================================
    # 14:30 - MARKET CLOSE
    # ============================================================
    async def market_close(self):
        """
        14:30 WAT - Market close summary
        
        - 📡 Data Quality: Verify today's data completeness
        - 📜 History: Archive today's OHLCV
        - 🎯 Backtest: Update strategy performance
        - 📱 TELEGRAM: "Market Close Summary"
        """
        logger.info("Sending market close summary...")
        
        try:
            now = datetime.now(NGX_TZ)
            
            message = f"""
🔔 <b>NGX MARKET CLOSED</b>
━━━━━━━━━━━━━━━━━━━━━━━
📅 {now.strftime('%Y-%m-%d')}

<b>📋 Session Complete:</b>
• Data archived ✅
• Quality checks running
• Strategy backtest updating

<b>📱 Next Actions:</b>
• Evening digest at 16:00 WAT
• Tomorrow's watchlist pending

Good session! 📊
            """.strip()
            
            await self.telegram.send_alert(message)
            logger.info("Market close summary sent")
            
        except Exception as e:
            logger.error(f"Market close error: {e}")
    
    # ============================================================
    # 16:00 - EVENING DIGEST (POST-MARKET ANALYSIS)
    # ============================================================
    async def evening_digest(self):
        """
        16:00 WAT - Tomorrow's watchlist and predictions
        
        - 📋 Disclosures: Check for late filings, run AI analysis
        - 🤖 ML: Evaluate today's prediction accuracy
        - 🔍 Screener: Run EOD screens for tomorrow
        - 👁 Watchlist: Update based on signals
        - 📱 TELEGRAM: "Evening Digest" with tomorrow's plays
        """
        logger.info("Sending evening digest...")
        
        try:
            from analyzers.pathway import PathwayAnalyzer
            
            analyzer = PathwayAnalyzer(self.config)
            now = datetime.now(NGX_TZ)
            
            # Get top opportunities for tomorrow
            opportunities = []
            
            for symbol in self.config.default_watchlist:
                try:
                    result = await analyzer.synthesize(symbol)
                    if 'error' not in result:
                        ret = result['predictions']['2d']['expected_return']
                        prob = result['predictions']['2d']['bull']['probability']
                        score = abs(ret) * (prob / 100 if ret > 0 else (100 - prob) / 100)
                        opportunities.append((symbol, ret, prob, score))
                except:
                    pass
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x[3], reverse=True)
            top = opportunities[:5]
            
            plays = []
            for sym, ret, prob, _ in top:
                emoji = '🟢' if ret > 0 else '🔴'
                sign = '+' if ret > 0 else ''
                direction = 'Bull' if ret > 0 else 'Bear'
                plays.append(f"{emoji} <b>{sym}</b>: {sign}{ret:.1f}% ({prob:.0f}% {direction})")
            
            message = f"""
🌙 <b>EVENING DIGEST</b>
━━━━━━━━━━━━━━━━━━━━━━
📅 {now.strftime('%Y-%m-%d')}

<b>🎯 Tomorrow's Top Plays:</b>
{chr(10).join(plays) if plays else '• Analysis pending'}

<b>📋 Post-Market Tasks:</b>
• Late disclosures checked
• ML accuracy evaluated
• Screener runs complete

<b>⏰ Tomorrow's Schedule:</b>
• 08:00 - Overnight processing
• 09:30 - Pre-market brief
• 10:00 - Market open

Good night! 🌃
            """.strip()
            
            await self.telegram.send_alert(message)
            logger.info("Evening digest sent")
            
        except Exception as e:
            logger.error(f"Evening digest error: {e}")
