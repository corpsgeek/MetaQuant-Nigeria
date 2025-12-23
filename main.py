#!/usr/bin/env python3
"""
MetaQuant Nigeria - Stock Screener MVP
A local-first desktop stock screener with fundamental filters,
portfolio tracking, and AI-powered insights.
"""

import sys
import os
import logging

# Load environment variables FIRST (before any other imports that need them)
from dotenv import load_dotenv
load_dotenv()

# Configure logging - suppress noisy tvDatafeed errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress tvDatafeed's noisy internal logging
logging.getLogger('tvDatafeed').setLevel(logging.CRITICAL)
logging.getLogger('tvDatafeed.main').setLevel(logging.CRITICAL)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.app import MetaQuantApp
from src.database.db_manager import DatabaseManager


def main():
    """Main entry point for the application."""
    # Initialize database
    db = DatabaseManager()
    db.initialize()
    
    # Launch GUI
    app = MetaQuantApp(db)
    app.run()


if __name__ == "__main__":
    main()

