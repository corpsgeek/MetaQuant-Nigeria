"""
NGX (Nigerian Exchange) data collector.
Scrapes equities price list and corporate disclosures from ngxgroup.com.
"""

import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd


logger = logging.getLogger(__name__)


class NGXCollector:
    """
    Collects market data from the Nigerian Exchange Group website.
    
    Data sources:
    - Equities Price List: https://ngxgroup.com/exchange/data/equities-price-list/
    - Corporate Disclosures: https://ngxgroup.com/exchange/data/corporate-disclosures/
    
    Notes:
    - Market hours: 10:00am - 2:30pm GMT+1
    - Data has ~30 minute delay from website
    """
    
    BASE_URL = "https://ngxgroup.com/exchange/data"
    PRICE_LIST_URL = f"{BASE_URL}/equities-price-list/"
    DISCLOSURES_URL = f"{BASE_URL}/corporate-disclosures/"
    
    # Request headers to mimic browser
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    # Market timing
    MARKET_OPEN_HOUR = 10  # 10:00 AM GMT+1
    MARKET_CLOSE_HOUR = 14  # 2:30 PM GMT+1
    MARKET_CLOSE_MINUTE = 30
    
    def __init__(self, request_delay: float = 1.0):
        """
        Initialize NGX collector.
        
        Args:
            request_delay: Delay between requests in seconds (be nice to the server)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.request_delay = request_delay
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[BeautifulSoup]:
        """Make an HTTP request and return parsed HTML."""
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_equities_price_list(self, page: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch equities price list from NGX.
        
        Args:
            page: Page number for pagination
            
        Returns:
            List of stock dictionaries with price data
        """
        stocks = []
        
        # NGX uses AJAX for pagination, try to find the API endpoint
        url = self.PRICE_LIST_URL
        if page > 1:
            url = f"{url}page/{page}/"
        
        soup = self._make_request(url)
        if soup is None:
            return stocks
        
        # Find the price list table
        # The table structure may vary, so we look for common patterns
        table = soup.find('table', class_=re.compile(r'.*price.*|.*equity.*', re.I))
        
        if table is None:
            # Try finding any data table
            table = soup.find('table', {'id': re.compile(r'.*price.*|.*data.*', re.I)})
        
        if table is None:
            # Look for div-based layouts
            cards = soup.find_all('div', class_=re.compile(r'.*stock.*|.*equity.*', re.I))
            for card in cards:
                stock_data = self._parse_stock_card(card)
                if stock_data:
                    stocks.append(stock_data)
            
            if stocks:
                return stocks
            
            logger.warning("Could not find price list table on page")
            return stocks
        
        # Parse table rows
        rows = table.find_all('tr')
        headers = []
        
        for i, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            
            if i == 0:
                # Header row
                headers = [self._clean_text(cell.get_text()) for cell in cells]
                continue
            
            if len(cells) < 3:
                continue
            
            # Parse data row
            stock_data = self._parse_price_row(cells, headers)
            if stock_data:
                stocks.append(stock_data)
        
        logger.info(f"Fetched {len(stocks)} stocks from page {page}")
        return stocks
    
    def get_all_equities(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch all equities across multiple pages.
        
        Args:
            max_pages: Maximum number of pages to fetch
            
        Returns:
            Combined list of all stocks
        """
        all_stocks = []
        
        for page in range(1, max_pages + 1):
            stocks = self.get_equities_price_list(page)
            
            if not stocks:
                # No more data, stop pagination
                break
            
            all_stocks.extend(stocks)
            logger.info(f"Total stocks fetched: {len(all_stocks)}")
        
        return all_stocks
    
    def _parse_price_row(self, cells: List, headers: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a single row from the price table."""
        try:
            data = {}
            
            for i, cell in enumerate(cells):
                if i < len(headers):
                    key = headers[i].lower().replace(' ', '_')
                    value = self._clean_text(cell.get_text())
                    data[key] = value
            
            # Map to standard format
            stock = {
                'symbol': data.get('symbol', data.get('code', '')).strip().upper(),
                'name': data.get('company', data.get('name', data.get('security', ''))),
                'last_price': self._parse_price(data.get('last_price', data.get('close', data.get('price', '0')))),
                'prev_close': self._parse_price(data.get('previous_close', data.get('prev_close', '0'))),
                'change': self._parse_price(data.get('change', '0')),
                'change_percent': self._parse_percent(data.get('change_%', data.get('change_percent', data.get('%_change', '0')))),
                'volume': self._parse_int(data.get('volume', data.get('deals_volume', '0'))),
                'value': self._parse_price(data.get('value', data.get('deals_value', '0'))),
                'trades': self._parse_int(data.get('trades', data.get('deals', '0'))),
                'high': self._parse_price(data.get('high', '0')),
                'low': self._parse_price(data.get('low', '0')),
                'open': self._parse_price(data.get('open', '0')),
                'sector': data.get('sector', ''),
                'last_updated': datetime.now(),
            }
            
            # Skip if no symbol
            if not stock['symbol']:
                return None
            
            return stock
            
        except Exception as e:
            logger.debug(f"Error parsing price row: {e}")
            return None
    
    def _parse_stock_card(self, card) -> Optional[Dict[str, Any]]:
        """Parse stock data from a card/div layout."""
        try:
            # Extract symbol and name
            symbol_elem = card.find(class_=re.compile(r'.*symbol.*|.*code.*', re.I))
            name_elem = card.find(class_=re.compile(r'.*name.*|.*company.*', re.I))
            price_elem = card.find(class_=re.compile(r'.*price.*', re.I))
            change_elem = card.find(class_=re.compile(r'.*change.*', re.I))
            
            symbol = symbol_elem.get_text().strip() if symbol_elem else ''
            
            if not symbol:
                return None
            
            return {
                'symbol': symbol.upper(),
                'name': name_elem.get_text().strip() if name_elem else symbol,
                'last_price': self._parse_price(price_elem.get_text() if price_elem else '0'),
                'change_percent': self._parse_percent(change_elem.get_text() if change_elem else '0'),
                'last_updated': datetime.now(),
            }
            
        except Exception as e:
            logger.debug(f"Error parsing stock card: {e}")
            return None
    
    def get_corporate_disclosures(
        self, 
        page: int = 1,
        stock_symbol: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch corporate disclosures from NGX.
        
        Args:
            page: Page number
            stock_symbol: Filter by stock symbol
            from_date: Filter from date
            to_date: Filter to date
            
        Returns:
            List of disclosure dictionaries
        """
        disclosures = []
        
        url = self.DISCLOSURES_URL
        if page > 1:
            url = f"{url}page/{page}/"
        
        soup = self._make_request(url)
        if soup is None:
            return disclosures
        
        # Find disclosure entries
        # Look for common disclosure patterns
        entries = soup.find_all('article') or soup.find_all('div', class_=re.compile(r'.*disclosure.*|.*post.*', re.I))
        
        for entry in entries:
            disclosure = self._parse_disclosure_entry(entry)
            if disclosure:
                # Apply filters
                if stock_symbol and disclosure.get('symbol', '').upper() != stock_symbol.upper():
                    continue
                if from_date and disclosure.get('date') and disclosure['date'] < from_date:
                    continue
                if to_date and disclosure.get('date') and disclosure['date'] > to_date:
                    continue
                
                disclosures.append(disclosure)
        
        logger.info(f"Fetched {len(disclosures)} disclosures from page {page}")
        return disclosures
    
    def _parse_disclosure_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse a single disclosure entry."""
        try:
            title_elem = entry.find(['h2', 'h3', 'h4']) or entry.find(class_=re.compile(r'.*title.*', re.I))
            date_elem = entry.find(class_=re.compile(r'.*date.*|.*time.*', re.I))
            link_elem = entry.find('a', href=True)
            content_elem = entry.find(class_=re.compile(r'.*content.*|.*excerpt.*|.*summary.*', re.I))
            
            title = title_elem.get_text().strip() if title_elem else ''
            
            if not title:
                return None
            
            # Try to extract stock symbol from title
            symbol_match = re.search(r'\b([A-Z]{2,10})\b', title)
            symbol = symbol_match.group(1) if symbol_match else ''
            
            # Parse date
            date_str = date_elem.get_text().strip() if date_elem else ''
            disclosure_date = self._parse_date(date_str)
            
            return {
                'title': title,
                'symbol': symbol,
                'date': disclosure_date,
                'content': content_elem.get_text().strip() if content_elem else '',
                'url': link_elem['href'] if link_elem else '',
                'disclosure_type': self._classify_disclosure(title),
            }
            
        except Exception as e:
            logger.debug(f"Error parsing disclosure entry: {e}")
            return None
    
    def _classify_disclosure(self, title: str) -> str:
        """Classify disclosure type based on title."""
        title_lower = title.lower()
        
        if any(term in title_lower for term in ['dividend', 'interim', 'final']):
            return 'DIVIDEND'
        elif any(term in title_lower for term in ['earnings', 'results', 'profit', 'loss']):
            return 'EARNINGS'
        elif any(term in title_lower for term in ['agm', 'meeting', 'egm']):
            return 'MEETING'
        elif any(term in title_lower for term in ['board', 'director', 'appointment']):
            return 'BOARD_CHANGES'
        elif any(term in title_lower for term in ['acquisition', 'merger', 'takeover']):
            return 'M&A'
        elif any(term in title_lower for term in ['rights', 'offer', 'issue']):
            return 'CAPITAL_RAISE'
        else:
            return 'GENERAL'
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ''
        return ' '.join(text.strip().split())
    
    @staticmethod
    def _parse_price(value: str) -> Optional[Decimal]:
        """Parse price string to Decimal."""
        if not value:
            return None
        try:
            # Remove currency symbols and commas
            clean = re.sub(r'[₦$,\s]', '', str(value))
            clean = clean.replace('NGN', '').strip()
            if clean in ('', '-', 'N/A', 'n/a'):
                return None
            return Decimal(clean)
        except:
            return None
    
    @staticmethod
    def _parse_percent(value: str) -> Optional[Decimal]:
        """Parse percentage string to Decimal."""
        if not value:
            return None
        try:
            clean = re.sub(r'[%\s]', '', str(value))
            if clean in ('', '-', 'N/A', 'n/a'):
                return None
            return Decimal(clean)
        except:
            return None
    
    @staticmethod
    def _parse_int(value: str) -> Optional[int]:
        """Parse integer string."""
        if not value:
            return None
        try:
            clean = re.sub(r'[,\s]', '', str(value))
            if clean in ('', '-', 'N/A', 'n/a'):
                return None
            return int(float(clean))
        except:
            return None
    
    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def is_market_open(self) -> bool:
        """Check if the NGX market is currently open."""
        now = datetime.now()
        
        # Check if weekday (Mon-Fri)
        if now.weekday() >= 5:
            return False
        
        # Check market hours (10:00 - 14:30 GMT+1)
        current_hour = now.hour
        current_minute = now.minute
        
        if current_hour < self.MARKET_OPEN_HOUR:
            return False
        if current_hour > self.MARKET_CLOSE_HOUR:
            return False
        if current_hour == self.MARKET_CLOSE_HOUR and current_minute > self.MARKET_CLOSE_MINUTE:
            return False
        
        return True
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        return {
            'is_open': self.is_market_open(),
            'open_time': f"{self.MARKET_OPEN_HOUR:02d}:00",
            'close_time': f"{self.MARKET_CLOSE_HOUR:02d}:{self.MARKET_CLOSE_MINUTE:02d}",
            'timezone': 'GMT+1',
            'data_delay_minutes': 30,
        }


# Convenience function
def fetch_ngx_stocks() -> List[Dict[str, Any]]:
    """Quick helper to fetch all NGX stocks."""
    collector = NGXCollector()
    return collector.get_all_equities()
