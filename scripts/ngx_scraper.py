"""
NGX Website Scraper
Scrapes real stock data from the Nigerian Exchange website using Selenium.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from src.database.db_manager import DatabaseManager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NGXScraper:
    """
    Scrapes stock data from NGX website.
    Uses headless Chrome for JavaScript rendering.
    """
    
    PRICE_LIST_URL = "https://ngxgroup.com/exchange/data/equities-price-list/"
    
    def __init__(self, headless: bool = True):
        """Initialize the scraper."""
        self.headless = headless
        self.driver = None
        self.db = DatabaseManager()
        self.db.initialize()
    
    def _init_driver(self):
        """Initialize Chrome WebDriver."""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("Chrome WebDriver initialized")
    
    def _close_driver(self):
        """Close WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_price_list(self) -> List[Dict]:
        """
        Scrape the equities price list from NGX.
        
        Returns:
            List of stock dictionaries with price data
        """
        if not self.driver:
            self._init_driver()
        
        stocks = []
        
        try:
            logger.info(f"Loading {self.PRICE_LIST_URL}")
            self.driver.get(self.PRICE_LIST_URL)
            
            # Wait for table to load (JavaScript rendered)
            wait = WebDriverWait(self.driver, 30)
            
            # Wait for data table
            try:
                table = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.dataTable, table#equities-table, .price-table"))
                )
                logger.info("Found data table")
            except:
                # Try alternative selectors
                logger.info("Trying alternative table selectors...")
                tables = self.driver.find_elements(By.TAG_NAME, "table")
                logger.info(f"Found {len(tables)} tables on page")
                
                if not tables:
                    # Check for data in different formats
                    logger.info("Looking for stock data in other formats...")
                    
                    # Sometimes data is in divs
                    stock_elements = self.driver.find_elements(By.CSS_SELECTOR, "[class*='stock'], [class*='equity'], [data-symbol]")
                    logger.info(f"Found {len(stock_elements)} potential stock elements")
            
            # Give extra time for dynamic content
            time.sleep(5)
            
            # Get page source for debugging
            page_source = self.driver.page_source
            
            # Try to find table rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            logger.info(f"Found {len(rows)} table rows")
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 6:
                        # Parse row data - format varies by page
                        stock = {
                            'symbol': cells[0].text.strip(),
                            'name': cells[1].text.strip() if len(cells) > 1 else '',
                            'last_price': self._parse_price(cells[2].text if len(cells) > 2 else '0'),
                            'change': self._parse_price(cells[3].text if len(cells) > 3 else '0'),
                            'volume': self._parse_volume(cells[4].text if len(cells) > 4 else '0'),
                        }
                        if stock['symbol']:
                            stocks.append(stock)
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue
            
            logger.info(f"Scraped {len(stocks)} stocks from NGX")
            
        except Exception as e:
            logger.error(f"Error scraping NGX: {e}")
            # Save page source for debugging
            if self.driver:
                with open("/tmp/ngx_page.html", "w") as f:
                    f.write(self.driver.page_source)
                logger.info("Saved page source to /tmp/ngx_page.html for debugging")
        
        return stocks
    
    def _parse_price(self, text: str) -> float:
        """Parse price string to float."""
        try:
            # Remove currency symbols, commas
            clean = text.replace('₦', '').replace(',', '').replace('NGN', '').strip()
            return float(clean) if clean else 0.0
        except:
            return 0.0
    
    def _parse_volume(self, text: str) -> int:
        """Parse volume string to int."""
        try:
            clean = text.replace(',', '').replace('M', '000000').replace('K', '000').strip()
            return int(float(clean)) if clean else 0
        except:
            return 0
    
    def update_database(self, stocks: List[Dict]) -> int:
        """
        Update database with scraped stock data.
        
        Returns:
            Number of stocks updated
        """
        count = 0
        today = datetime.now().strftime('%Y-%m-%d')
        
        for stock in stocks:
            try:
                # Get existing stock or skip
                db_stock = self.db.get_stock(stock['symbol'])
                if not db_stock:
                    logger.debug(f"Stock {stock['symbol']} not in universe, skipping")
                    continue
                
                # Update stock price
                self.db.upsert_stock({
                    'symbol': stock['symbol'],
                    'name': db_stock['name'],  
                    'sector': db_stock.get('sector'),
                    'last_price': stock['last_price'],
                    'prev_close': db_stock.get('last_price'),
                    'change_percent': stock.get('change', 0),
                    'volume': stock.get('volume', 0),
                    'is_active': True,
                    'last_updated': datetime.now(),
                })
                
                # Insert daily price
                if stock['last_price'] > 0:
                    self.db.insert_daily_price(
                        stock_id=db_stock['id'],
                        date=today,
                        ohlcv={
                            'open': stock['last_price'],  # Approximate
                            'high': stock['last_price'],
                            'low': stock['last_price'],
                            'close': stock['last_price'],
                            'volume': stock.get('volume', 0),
                        }
                    )
                
                count += 1
                
            except Exception as e:
                logger.error(f"Error updating {stock.get('symbol')}: {e}")
        
        logger.info(f"Updated {count} stocks in database")
        return count
    
    def run(self) -> int:
        """
        Run the full scraping process.
        
        Returns:
            Number of stocks updated
        """
        try:
            self._init_driver()
            stocks = self.scrape_price_list()
            
            if stocks:
                count = self.update_database(stocks)
                
                # Also update market snapshot
                self._create_snapshot(stocks)
                
                return count
            else:
                logger.warning("No stocks scraped")
                return 0
        finally:
            self._close_driver()
            self.db.close()
    
    def _create_snapshot(self, stocks: List[Dict]):
        """Create market snapshot from scraped data."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        gainers = [s for s in stocks if s.get('change', 0) > 0]
        losers = [s for s in stocks if s.get('change', 0) < 0]
        gainers.sort(key=lambda x: -x.get('change', 0))
        losers.sort(key=lambda x: x.get('change', 0))
        
        snapshot = {
            'total_volume': sum(s.get('volume', 0) for s in stocks),
            'gainers_count': len(gainers),
            'losers_count': len(losers),
            'unchanged_count': len(stocks) - len(gainers) - len(losers),
            'top_gainer_symbol': gainers[0]['symbol'] if gainers else None,
            'top_gainer_change': gainers[0].get('change') if gainers else None,
            'top_loser_symbol': losers[0]['symbol'] if losers else None,
            'top_loser_change': losers[0].get('change') if losers else None,
        }
        
        self.db.save_market_snapshot(today, snapshot)
        logger.info(f"Created market snapshot for {today}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NGX Website Scraper")
    parser.add_argument("--visible", action="store_true", help="Run with visible browser (for debugging)")
    
    args = parser.parse_args()
    
    scraper = NGXScraper(headless=not args.visible)
    
    try:
        count = scraper.run()
        print(f"\n✅ Scraped and updated {count} stocks!")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
