"""
IMDT Data Scraper
Scrapes real-time NGX market data from InfoWare IMDT platform using Playwright.
Includes securities prices and order book data.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from src.database.db_manager import DatabaseManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IMDTScraper:
    """
    Scrapes NGX market data from InfoWare IMDT platform.
    Uses Playwright for browser automation with authenticated access.
    """
    
    LOGIN_URL = "https://idia.infowarelimited.com/IMDT/login?returnUrl="
    DASHBOARD_URL = "https://svcs.infowarelimited.com/IMDT/"
    
    def __init__(
        self,
        email: str,
        password: str,
        headless: bool = True,
        db_path: Optional[str] = None
    ):
        """
        Initialize the IMDT scraper.
        
        Args:
            email: IMDT login email
            password: IMDT login password
            headless: Run browser in headless mode
            db_path: Optional database path
        """
        self.email = email
        self.password = password
        self.headless = headless
        self.db = DatabaseManager(db_path)
        self.db.initialize()
        
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
        logger.info("IMDTScraper initialized")
    
    def connect(self) -> bool:
        """Launch browser and login to IMDT."""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            
            # Login flow
            logger.info("Logging in to IMDT...")
            self.page.goto(self.LOGIN_URL)
            
            # Fill login form
            self.page.get_by_role("textbox", name="Email").fill(self.email)
            self.page.get_by_role("textbox", name="Password").fill(self.password)
            self.page.get_by_role("button", name="Login").click()
            
            # Wait for redirect and navigate to dashboard
            self.page.wait_for_load_state("networkidle")
            time.sleep(2)
            
            self.page.goto(self.DASHBOARD_URL)
            self.page.wait_for_load_state("networkidle")
            
            logger.info("✅ Successfully connected to IMDT")
            return True
            
        except Exception as e:
            logger.error(f"❌ IMDT connection failed: {e}")
            return False
    
    def navigate_to_equities(self) -> bool:
        """Navigate to the PREMIUM EQTY board."""
        try:
            # Click PREMIUM tab
            self.page.get_by_text("PREMIUM").first.click()
            time.sleep(1)
            
            # Select EQTY from dropdown
            # The dropdown has a dynamic ID, so we use a more flexible selector
            dropdown = self.page.locator(".rz-dropdown-trigger").first
            dropdown.click()
            time.sleep(0.5)
            
            self.page.get_by_role("option", name="EQTY").click()
            time.sleep(2)
            
            logger.info("✅ Navigated to EQTY board")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to navigate to equities: {e}")
            return False
    
    def scrape_securities_table(self) -> List[Dict[str, Any]]:
        """
        Scrape all securities from the EQTY table.
        
        Returns:
            List of security dictionaries with price data
        """
        securities = []
        
        try:
            # Wait for table to be fully loaded
            self.page.wait_for_selector("table", timeout=10000)
            time.sleep(2)
            
            # Get all table rows
            rows = self.page.locator("table tbody tr").all()
            logger.info(f"Found {len(rows)} securities in table")
            
            for i, row in enumerate(rows):
                try:
                    cells = row.locator("td").all()
                    if len(cells) < 10:
                        continue
                    
                    # Extract cell values
                    security = {
                        'symbol': self._get_cell_text(cells[0]),           # SecurityName
                        'board': self._get_cell_text(cells[1]),            # Board
                        'last_price': self._parse_price(cells[2]),         # LstPrice
                        'wa_price': self._parse_price(cells[3]),           # WAPrice (weighted avg)
                        'open': self._parse_price(cells[4]),               # Open
                        'ref_price': self._parse_price(cells[5]),          # RefPrice
                        'change': self._parse_price(cells[6]),             # Change
                        'days_range': self._get_cell_text(cells[7]),       # Day's Range
                        'closing_price': self._parse_price(cells[8]),      # ClosingPrice
                        'volume': self._parse_volume(cells[9]),            # Volume
                        'trades': self._parse_int(cells[10]) if len(cells) > 10 else 0,  # #OfTrades
                        'weeks_range': self._get_cell_text(cells[11]) if len(cells) > 11 else '',  # Weeks Range
                    }
                    
                    if security['symbol']:
                        securities.append(security)
                        if (i + 1) % 20 == 0:
                            logger.info(f"Processed {i + 1} securities...")
                            
                except Exception as e:
                    logger.debug(f"Error parsing row {i}: {e}")
                    continue
            
            logger.info(f"✅ Scraped {len(securities)} securities")
            return securities
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape securities table: {e}")
            return []
    
    def scrape_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scrape order book data for a specific security.
        
        Args:
            symbol: Security symbol to fetch order book for
            
        Returns:
            Dictionary with bid/ask levels or None if failed
        """
        try:
            # Click on the security row
            self.page.get_by_text(symbol, exact=True).first.click()
            time.sleep(1.5)
            
            # Click on Price Book tab
            self.page.get_by_text(f"{symbol} Price Book", exact=False).click()
            time.sleep(1)
            
            # Get the Price Book tabpanel
            tabpanel = self.page.get_by_role("tabpanel", name="Price Book")
            
            # Extract bid/ask data from the table
            orderbook = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': [],
                'asks': []
            }
            
            # Get all rows in the order book table
            rows = tabpanel.locator("table tbody tr").all()
            
            for row in rows:
                cells = row.locator("td").all()
                if len(cells) >= 6:
                    # Format: Ord | Vol | Bid | Ask | Vol | Ord | Total
                    bid_data = {
                        'orders': self._parse_int(cells[0]),
                        'volume': self._parse_volume(cells[1]),
                        'price': self._parse_price(cells[2])
                    }
                    ask_data = {
                        'price': self._parse_price(cells[3]),
                        'volume': self._parse_volume(cells[4]),
                        'orders': self._parse_int(cells[5])
                    }
                    
                    if bid_data['price'] > 0:
                        orderbook['bids'].append(bid_data)
                    if ask_data['price'] > 0:
                        orderbook['asks'].append(ask_data)
            
            # Navigate back to main table
            self.page.keyboard.press("Escape")
            time.sleep(0.5)
            
            logger.info(f"✅ Scraped order book for {symbol}: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
            return orderbook
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape order book for {symbol}: {e}")
            return None
    
    def _get_cell_text(self, cell) -> str:
        """Get text content from a table cell."""
        try:
            return cell.inner_text().strip()
        except:
            return ""
    
    def _parse_price(self, cell) -> float:
        """Parse price value from cell."""
        try:
            text = cell.inner_text() if hasattr(cell, 'inner_text') else str(cell)
            text = text.strip().replace(',', '').replace('₦', '').replace('NGN', '')
            return float(text) if text else 0.0
        except:
            return 0.0
    
    def _parse_volume(self, cell) -> int:
        """Parse volume value from cell."""
        try:
            text = cell.inner_text() if hasattr(cell, 'inner_text') else str(cell)
            text = text.strip().replace(',', '')
            # Handle M/K suffixes
            if 'M' in text.upper():
                return int(float(text.upper().replace('M', '')) * 1_000_000)
            elif 'K' in text.upper():
                return int(float(text.upper().replace('K', '')) * 1_000)
            return int(float(text)) if text else 0
        except:
            return 0
    
    def _parse_int(self, cell) -> int:
        """Parse integer value from cell."""
        try:
            text = cell.inner_text() if hasattr(cell, 'inner_text') else str(cell)
            text = text.strip().replace(',', '')
            return int(float(text)) if text else 0
        except:
            return 0
    
    def update_database(self, securities: List[Dict[str, Any]]) -> int:
        """
        Update database with scraped securities data.
        
        Returns:
            Number of stocks updated
        """
        count = 0
        today = datetime.now().strftime('%Y-%m-%d')
        
        for sec in securities:
            try:
                symbol = sec['symbol']
                
                # Calculate change percent
                change_pct = 0
                if sec['ref_price'] and sec['ref_price'] > 0:
                    change_pct = round((sec['last_price'] - sec['ref_price']) / sec['ref_price'] * 100, 2)
                
                # Upsert stock
                stock_id = self.db.upsert_stock({
                    'symbol': symbol,
                    'name': symbol,  # IMDT doesn't give full name in table
                    'last_price': sec['last_price'],
                    'prev_close': sec['ref_price'],
                    'change_percent': change_pct,
                    'volume': sec['volume'],
                    'is_active': True,
                    'last_updated': datetime.now(),
                })
                
                # Insert daily price
                if sec['last_price'] > 0:
                    # Parse day's range for high/low
                    high = sec['last_price']
                    low = sec['last_price']
                    if sec['days_range'] and '-' in sec['days_range']:
                        try:
                            parts = sec['days_range'].split('-')
                            low = float(parts[0].strip().replace(',', ''))
                            high = float(parts[1].strip().replace(',', ''))
                        except:
                            pass
                    
                    self.db.insert_daily_price(
                        stock_id=stock_id,
                        date=today,
                        ohlcv={
                            'open': sec.get('open', sec['last_price']),
                            'high': high,
                            'low': low,
                            'close': sec['last_price'],
                            'volume': sec['volume'],
                        }
                    )
                
                count += 1
                
            except Exception as e:
                logger.error(f"Error updating {sec.get('symbol')}: {e}")
        
        logger.info(f"✅ Updated {count} stocks in database")
        return count
    
    def save_orderbook(self, orderbook: Dict[str, Any]) -> bool:
        """Save order book snapshot to database."""
        try:
            symbol = orderbook['symbol']
            stock = self.db.get_stock(symbol)
            if not stock:
                logger.warning(f"Stock {symbol} not found in database")
                return False
            
            stock_id = stock['id']
            
            # Prepare order book data for database
            ob_data = {'timestamp': orderbook['timestamp']}
            
            # Add bid levels (up to 5)
            for i, bid in enumerate(orderbook['bids'][:5], 1):
                ob_data[f'bid_price_{i}'] = bid['price']
                ob_data[f'bid_volume_{i}'] = bid['volume']
            
            # Add ask levels (up to 5)
            for i, ask in enumerate(orderbook['asks'][:5], 1):
                ob_data[f'ask_price_{i}'] = ask['price']
                ob_data[f'ask_volume_{i}'] = ask['volume']
            
            self.db.insert_orderbook_snapshot(stock_id, ob_data)
            return True
            
        except Exception as e:
            logger.error(f"Error saving order book: {e}")
            return False
    
    def create_market_snapshot(self, securities: List[Dict[str, Any]]):
        """Create daily market snapshot from scraped data."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        gainers = [s for s in securities if s.get('change', 0) > 0]
        losers = [s for s in securities if s.get('change', 0) < 0]
        gainers.sort(key=lambda x: -x.get('change', 0))
        losers.sort(key=lambda x: x.get('change', 0))
        
        total_trades = sum(s.get('trades', 0) for s in securities)
        
        snapshot = {
            'total_volume': sum(s.get('volume', 0) for s in securities),
            'total_trades': total_trades,
            'gainers_count': len(gainers),
            'losers_count': len(losers),
            'unchanged_count': len(securities) - len(gainers) - len(losers),
            'top_gainer_symbol': gainers[0]['symbol'] if gainers else None,
            'top_gainer_change': gainers[0].get('change') if gainers else None,
            'top_loser_symbol': losers[0]['symbol'] if losers else None,
            'top_loser_change': losers[0].get('change') if losers else None,
        }
        
        self.db.save_market_snapshot(today, snapshot)
        logger.info(f"✅ Created market snapshot for {today}")
    
    def run_update(self) -> int:
        """
        Run full update: connect, scrape, and update database.
        
        Returns:
            Number of securities updated
        """
        try:
            if not self.connect():
                return 0
            
            if not self.navigate_to_equities():
                return 0
            
            securities = self.scrape_securities_table()
            
            if securities:
                count = self.update_database(securities)
                self.create_market_snapshot(securities)
                return count
            else:
                logger.warning("No securities scraped")
                return 0
                
        finally:
            self.close()
    
    def close(self):
        """Clean up browser resources."""
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.db.close()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")


def main():
    """Main entry point for IMDT scraper."""
    parser = argparse.ArgumentParser(description="IMDT Market Data Scraper")
    parser.add_argument("--update", action="store_true", help="Fetch all securities prices")
    parser.add_argument("--orderbook", type=str, help="Fetch order book for specific symbol")
    parser.add_argument("--orderbook-all", action="store_true", help="Fetch order book for all securities")
    parser.add_argument("--visible", action="store_true", help="Run with visible browser")
    
    args = parser.parse_args()
    
    # Get credentials from environment
    IMDT_EMAIL = os.getenv("IMDT_EMAIL")
    IMDT_PASSWORD = os.getenv("IMDT_PASSWORD")
    
    if not IMDT_EMAIL or not IMDT_PASSWORD:
        logger.error("❌ IMDT_EMAIL and IMDT_PASSWORD must be set in environment variables")
        logger.info("Set them in .env file or export them:")
        logger.info("  export IMDT_EMAIL='your_email'")
        logger.info("  export IMDT_PASSWORD='your_password'")
        sys.exit(1)
    
    scraper = IMDTScraper(
        email=IMDT_EMAIL,
        password=IMDT_PASSWORD,
        headless=not args.visible
    )
    
    try:
        if args.update or (not args.orderbook and not args.orderbook_all):
            # Default action: update all securities
            count = scraper.run_update()
            print(f"\n✅ Updated {count} securities from IMDT!")
        
        if args.orderbook:
            # Fetch order book for specific symbol
            if scraper.connect() and scraper.navigate_to_equities():
                orderbook = scraper.scrape_orderbook(args.orderbook)
                if orderbook:
                    scraper.save_orderbook(orderbook)
                    print(f"\n✅ Order book saved for {args.orderbook}")
                else:
                    print(f"\n❌ Failed to fetch order book for {args.orderbook}")
        
        if args.orderbook_all:
            # Fetch order book for all securities
            if scraper.connect() and scraper.navigate_to_equities():
                securities = scraper.scrape_securities_table()
                for sec in securities:
                    orderbook = scraper.scrape_orderbook(sec['symbol'])
                    if orderbook:
                        scraper.save_orderbook(orderbook)
                    time.sleep(0.5)  # Rate limiting
                print(f"\n✅ Order books saved for {len(securities)} securities")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
