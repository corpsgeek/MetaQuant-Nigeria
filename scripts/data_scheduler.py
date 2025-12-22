#!/usr/bin/env python
"""
NGX Data Scheduler

Runs the market data loader automatically at scheduled times.
Can run as a background service or be triggered by macOS launchd.

Usage:
    # Run as continuous scheduler (keeps running)
    python scripts/data_scheduler.py
    
    # Run once (for launchd/cron)
    python scripts/data_scheduler.py --once
    
    # Test mode
    python scripts/data_scheduler.py --test
"""

import sys
import os
import logging
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

from scripts.load_market_data import load_market_data, show_market_snapshot

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NGX Market hours (WAT = GMT+1)
MARKET_CLOSE_HOUR = 14  # 2:30 PM WAT
MARKET_CLOSE_MINUTE = 30
DATA_LOAD_HOUR = 15  # Run at 3:00 PM WAT (30 min after close)
DATA_LOAD_MINUTE = 0


def is_trading_day() -> bool:
    """Check if today is a trading day (Mon-Fri, not a holiday)."""
    today = datetime.now()
    # Weekday: Monday=0, Sunday=6
    if today.weekday() >= 5:
        return False
    # TODO: Add Nigerian public holidays check
    return True


def run_data_load():
    """Execute the data load job."""
    logger.info("=" * 50)
    logger.info("Starting scheduled data load...")
    
    if not is_trading_day():
        logger.info("Not a trading day, skipping data load")
        return
    
    try:
        result = load_market_data()
        
        if result['success']:
            logger.info(f"✓ Data load successful: {result['stored']} stocks for {result['date']}")
        else:
            logger.error(f"✗ Data load failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error during scheduled data load: {e}")
    
    logger.info("=" * 50)


def run_scheduler():
    """Run the continuous scheduler."""
    if not SCHEDULE_AVAILABLE:
        logger.error("schedule package not installed. Run: pip install schedule")
        return
    
    logger.info(f"Starting NGX Data Scheduler...")
    logger.info(f"Scheduled to run daily at {DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d} (WAT)")
    
    # Schedule daily job at 3:00 PM
    schedule.every().monday.at(f"{DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d}").do(run_data_load)
    schedule.every().tuesday.at(f"{DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d}").do(run_data_load)
    schedule.every().wednesday.at(f"{DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d}").do(run_data_load)
    schedule.every().thursday.at(f"{DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d}").do(run_data_load)
    schedule.every().friday.at(f"{DATA_LOAD_HOUR:02d}:{DATA_LOAD_MINUTE:02d}").do(run_data_load)
    
    # Also run at startup if we missed today's load
    now = datetime.now()
    load_time = now.replace(hour=DATA_LOAD_HOUR, minute=DATA_LOAD_MINUTE, second=0)
    
    if is_trading_day() and now > load_time:
        logger.info("Checking if we need to catch up on today's data...")
        # Could add logic here to check if today's data is already loaded
    
    logger.info("Scheduler running. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


def backfill_missing_days(days: int = 7):
    """
    Try to backfill missing historical data.
    Note: TradingView only provides current day data, so this just loads today.
    """
    logger.info(f"Note: TradingView only provides current day data.")
    logger.info("Running single day load...")
    run_data_load()


def main():
    parser = argparse.ArgumentParser(description='NGX Data Scheduler')
    parser.add_argument('--once', action='store_true', 
                        help='Run once and exit (for cron/launchd)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode - run immediately')
    parser.add_argument('--backfill', type=int, default=0,
                        help='Try to backfill N days of missing data')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Test mode - running data load now")
        run_data_load()
        return
    
    if args.once:
        logger.info("Running one-time data load")
        run_data_load()
        return
    
    if args.backfill > 0:
        backfill_missing_days(args.backfill)
        return
    
    # Run continuous scheduler
    run_scheduler()


if __name__ == "__main__":
    main()
