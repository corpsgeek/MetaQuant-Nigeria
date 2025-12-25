# Scheduled Jobs for MetaQuant Daemon

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
        
    async def overnight_processing(self):
        """08:00 - ML training, disclosure scraping, PCA update."""
        logger.info("Running overnight processing...")
        
        try:
            # Scrape disclosures
            from analyzers.disclosures import scrape_and_analyze_disclosures
            disclosure_alerts = await scrape_and_analyze_disclosures(self.config)
            
            # Send high-impact disclosures
            for alert in disclosure_alerts:
                await self.telegram.send_alert(alert)
            
            # Train ML models
            from analyzers.ml_signals import retrain_models
            await retrain_models(self.config)
            
            # Update PCA factors
            from analyzers.pca import update_pca_factors
            await update_pca_factors(self.config)
            
            logger.info("Overnight processing complete")
            
        except Exception as e:
            logger.error(f"Overnight processing error: {e}")
    
    async def pre_market_brief(self):
        """09:30 - Send pre-market briefing."""
        logger.info("Sending pre-market briefing...")
        
        try:
            from analyzers.market_intel import generate_pre_market_brief
            brief = await generate_pre_market_brief(self.config)
            await self.telegram.send_alert(brief)
            
        except Exception as e:
            logger.error(f"Pre-market brief error: {e}")
    
    async def market_open(self):
        """10:00 - Market open notification."""
        logger.info("Market open alert...")
        
        message = """
🔔 <b>NGX Market Open</b>
━━━━━━━━━━━━━━━━━━━━

The Nigerian Stock Exchange is now open.
Trading hours: 10:00 - 14:30 WAT

Use /summary for real-time analysis.
        """
        await self.telegram.send_alert(message.strip())
    
    async def intraday_scan(self):
        """Every 15 min - Scan for flow and anomaly signals."""
        logger.info("Running intraday scan...")
        
        try:
            alerts = []
            
            # Flow analysis
            from analyzers.flow import scan_all_flow
            flow_alerts = await scan_all_flow(self.config)
            alerts.extend(flow_alerts)
            
            # Anomaly detection
            from analyzers.ml_signals import detect_anomalies
            anomaly_alerts = await detect_anomalies(self.config)
            alerts.extend(anomaly_alerts)
            
            # Send significant alerts only
            for alert in alerts:
                await self.telegram.send_alert(alert)
                
        except Exception as e:
            logger.error(f"Intraday scan error: {e}")
    
    async def midday_synthesis(self):
        """12:00 - Full pathway refresh for watchlist."""
        logger.info("Running midday synthesis...")
        
        try:
            from analyzers.pathway import generate_watchlist_pathways
            synthesis = await generate_watchlist_pathways(self.config)
            await self.telegram.send_alert(synthesis)
            
        except Exception as e:
            logger.error(f"Midday synthesis error: {e}")
    
    async def pre_close(self):
        """14:00 - Pre-close positioning signals."""
        logger.info("Sending pre-close signals...")
        
        try:
            from analyzers.pathway import generate_close_signals
            signals = await generate_close_signals(self.config)
            await self.telegram.send_alert(signals)
            
        except Exception as e:
            logger.error(f"Pre-close error: {e}")
    
    async def market_close(self):
        """14:30 - Market close summary."""
        logger.info("Sending market close summary...")
        
        try:
            from analyzers.market_intel import generate_eod_summary
            summary = await generate_eod_summary(self.config)
            await self.telegram.send_alert(summary)
            
            message = """
🔔 <b>NGX Market Closed</b>
━━━━━━━━━━━━━━━━━━━━

Trading has ended for today.
Evening digest at 16:00 WAT.
            """
            await self.telegram.send_alert(message.strip())
            
        except Exception as e:
            logger.error(f"Market close error: {e}")
    
    async def evening_digest(self):
        """16:00 - Tomorrow's watchlist and predictions."""
        logger.info("Sending evening digest...")
        
        try:
            from analyzers.signals import generate_evening_digest
            digest = await generate_evening_digest(self.config)
            await self.telegram.send_alert(digest)
            
        except Exception as e:
            logger.error(f"Evening digest error: {e}")
