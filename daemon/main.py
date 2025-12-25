# MetaQuant Telegram Daemon
# Headless daemon for NGX market analysis with Telegram alerts

import os
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from config import Config
from telegram_bot.bot import TelegramBot
from scheduler.jobs import ScheduledJobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('daemon.log')
    ]
)
logger = logging.getLogger(__name__)

# NGX timezone
NGX_TZ = pytz.timezone('Africa/Lagos')


class MetaQuantDaemon:
    """Main daemon coordinator for MetaQuant alerts."""
    
    def __init__(self):
        self.config = Config()
        self.scheduler = AsyncIOScheduler(timezone=NGX_TZ)
        self.telegram_bot = TelegramBot(self.config)
        self.jobs = ScheduledJobs(self.config, self.telegram_bot)
        
    def setup_schedule(self):
        """Configure all scheduled jobs based on NGX market hours."""
        
        # Overnight processing (08:00 WAT)
        self.scheduler.add_job(
            self.jobs.overnight_processing,
            CronTrigger(hour=8, minute=0, timezone=NGX_TZ),
            id='overnight_processing',
            name='Overnight ML Training & Disclosure Scrape'
        )
        
        # Pre-market briefing (09:30 WAT)
        self.scheduler.add_job(
            self.jobs.pre_market_brief,
            CronTrigger(hour=9, minute=30, timezone=NGX_TZ),
            id='pre_market_brief',
            name='Pre-Market Briefing'
        )
        
        # Market open (10:00 WAT)
        self.scheduler.add_job(
            self.jobs.market_open,
            CronTrigger(hour=10, minute=0, timezone=NGX_TZ),
            id='market_open',
            name='Market Open Alert'
        )
        
        # Market updates (every 30 min from 10:30 to 13:30)
        self.scheduler.add_job(
            self.jobs.market_open,  # Same as market_open for updates
            CronTrigger(minute='30', hour='10-13', timezone=NGX_TZ),
            id='market_update_30',
            name='Market Update (30min)'
        )
        
        # Midday synthesis (hourly at 11:00, 12:00, 13:00)
        self.scheduler.add_job(
            self.jobs.midday_synthesis,
            CronTrigger(minute=0, hour='11-13', timezone=NGX_TZ),
            id='midday_synthesis',
            name='Midday Synthesis (Hourly)'
        )
        
        # Pre-close positioning (14:00 WAT)
        self.scheduler.add_job(
            self.jobs.pre_close,
            CronTrigger(hour=14, minute=0, timezone=NGX_TZ),
            id='pre_close',
            name='Pre-Close Positioning'
        )
        
        # Market close (14:30 WAT)
        self.scheduler.add_job(
            self.jobs.market_close,
            CronTrigger(hour=14, minute=30, timezone=NGX_TZ),
            id='market_close',
            name='Market Close Summary'
        )
        
        # Evening digest (16:00 WAT)
        self.scheduler.add_job(
            self.jobs.evening_digest,
            CronTrigger(hour=16, minute=0, timezone=NGX_TZ),
            id='evening_digest',
            name='Evening Digest'
        )
        
        logger.info(f"Scheduled {len(self.scheduler.get_jobs())} jobs")
        
    async def run(self):
        """Start the daemon."""
        logger.info("=" * 50)
        logger.info("MetaQuant Telegram Daemon Starting")
        logger.info(f"Timezone: {NGX_TZ}")
        logger.info(f"Current time: {datetime.now(NGX_TZ)}")
        logger.info("=" * 50)
        
        # Setup scheduler
        self.setup_schedule()
        self.scheduler.start()
        
        # Start Telegram bot
        await self.telegram_bot.start()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down daemon...")
            self.scheduler.shutdown()
            await self.telegram_bot.stop()


async def main():
    daemon = MetaQuantDaemon()
    await daemon.run()


if __name__ == '__main__':
    asyncio.run(main())
