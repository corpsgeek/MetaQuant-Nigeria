"""
Corporate Disclosure Scraper for MetaQuant Nigeria.
Fetches disclosure announcements from NGX website.
"""

import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re
import time

logger = logging.getLogger(__name__)

# Base URLs
NGX_DISCLOSURES_URL = "https://ngxgroup.com/exchange/data/corporate-disclosures/"
NGX_DOC_LIB_BASE = "https://doclib.ngxgroup.com"


class DisclosureScraper:
    """Scrapes corporate disclosures from NGX website."""
    
    def __init__(self, db):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        self._init_table()
    
    def _init_table(self):
        """Initialize database table for disclosures."""
        # Check if table has correct schema by checking columns
        try:
            result = self.db.conn.execute("""
                SELECT company_symbol FROM corporate_disclosures LIMIT 1
            """).fetchone()
        except:
            # Column doesn't exist - drop and recreate with correct schema
            logger.info("Recreating corporate_disclosures table with correct schema...")
            try:
                self.db.conn.execute("DROP TABLE IF EXISTS corporate_disclosures")
                self.db.conn.execute("DROP SEQUENCE IF EXISTS disclosure_id_seq")
            except:
                pass
        
        # Create sequence for auto-increment ID
        try:
            self.db.conn.execute("CREATE SEQUENCE IF NOT EXISTS disclosure_id_seq START 1")
        except:
            pass
        
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_disclosures (
                id INTEGER PRIMARY KEY DEFAULT nextval('disclosure_id_seq'),
                company_symbol TEXT,
                company_name TEXT,
                title TEXT,
                date_submitted TEXT,
                pdf_url TEXT UNIQUE,
                pdf_text TEXT,
                ai_summary TEXT,
                impact_score INTEGER,
                impact_label TEXT,
                key_highlights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        """)
        self.db.conn.commit()
        logger.info("Corporate disclosures table initialized")
    
    def fetch_disclosures(self, limit: int = 100) -> List[Dict]:
        """
        Fetch latest disclosures from NGX website.
        
        Args:
            limit: Maximum number of disclosures to fetch
            
        Returns:
            List of disclosure dictionaries
        """
        disclosures = []
        
        try:
            logger.info(f"Fetching disclosures from NGX (limit={limit})...")
            
            # Try Selenium first for JavaScript-rendered content
            disclosures = self._fetch_with_selenium(limit)
            if disclosures:
                return disclosures
            
            # Fallback to requests (may not work if JS-rendered)
            response = self.session.get(NGX_DISCLOSURES_URL, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the correct table ID
            table = soup.find('table', {'id': 'latestdiclosuresLanding'})
            if not table:
                table = soup.find('table', {'id': 'company-disclosure-table'})
            if not table:
                table = soup.find('table', {'class': 'dataTable'})
            
            if not table:
                logger.warning("Could not find disclosures table on page - may need Selenium")
                return self._parse_disclosures_alternative(soup, limit)
            
            # Parse table rows
            tbody = table.find('tbody', {'id': 'landing_corp_disclosure'})
            if not tbody:
                tbody = table.find('tbody')
            rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]
            
            for row in rows[:limit]:
                try:
                    disclosure = self._parse_table_row(row)
                    if disclosure:
                        disclosures.append(disclosure)
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}")
                    continue
            
            logger.info(f"Fetched {len(disclosures)} disclosures")
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch disclosures page: {e}")
        except Exception as e:
            logger.error(f"Error parsing disclosures: {e}")
        
        return disclosures
    
    def _fetch_with_selenium(self, limit: int) -> List[Dict]:
        """Fetch disclosures using Selenium for JavaScript-rendered content."""
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.chrome.options import Options
            
            logger.info("Starting Selenium fetch...")
            
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=options)
            driver.get(NGX_DISCLOSURES_URL)
            logger.info("Page loaded, waiting for table...")
            
            # Wait for table to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "latestdiclosuresLanding"))
                )
                logger.info("Table found, waiting for data to load...")
            except Exception as e:
                logger.warning(f"Table not found after 15s: {e}")
                driver.quit()
                return []
            
            time.sleep(3)  # Extra wait for data to populate
            
            disclosures = []
            
            # Try finding rows
            rows = driver.find_elements(By.CSS_SELECTOR, "#landing_corp_disclosure tr")
            logger.info(f"Found {len(rows)} rows in table")
            
            if len(rows) == 0:
                # Try alternative selector
                rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                logger.info(f"Alternative selector found {len(rows)} rows")
            
            for row in rows[:limit]:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 3:
                        continue
                    
                    # Company
                    company_links = cells[0].find_elements(By.TAG_NAME, "a")
                    company_link = company_links[0] if company_links else None
                    company_name = cells[0].text.strip()
                    company_symbol = None
                    if company_link:
                        href = company_link.get_attribute('href')
                        match = re.search(r'symbol=(\w+)', href)
                        if match:
                            company_symbol = match.group(1)
                    
                    # Title and PDF
                    title_links = cells[1].find_elements(By.TAG_NAME, "a")
                    title_link = title_links[0] if title_links else None
                    title = cells[1].text.strip()
                    pdf_url = title_link.get_attribute('href') if title_link else ''
                    
                    # Date
                    date_submitted = cells[2].text.strip()
                    
                    if title and pdf_url:  # Only add if we have content
                        disclosures.append({
                            'company_symbol': company_symbol,
                            'company_name': company_name,
                            'title': title,
                            'date_submitted': date_submitted,
                            'pdf_url': pdf_url
                        })
                except Exception as e:
                    logger.debug(f"Row parse error: {e}")
                    continue
            
            driver.quit()
            logger.info(f"Fetched {len(disclosures)} disclosures via Selenium")
            return disclosures
            
        except ImportError as e:
            logger.info(f"Selenium not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"Selenium fetch failed: {e}")
            return []
    
    def _parse_table_row(self, row) -> Optional[Dict]:
        """Parse a single table row into disclosure dict."""
        cells = row.find_all('td')
        if len(cells) < 3:
            return None
        
        # Extract company info
        company_cell = cells[0]
        company_link = company_cell.find('a')
        company_name = company_link.text.strip() if company_link else company_cell.text.strip()
        
        # Extract symbol from link
        company_symbol = None
        if company_link and 'symbol=' in str(company_link.get('href', '')):
            match = re.search(r'symbol=(\w+)', company_link.get('href', ''))
            if match:
                company_symbol = match.group(1)
        
        # Extract title and PDF URL
        title_cell = cells[1]
        title_link = title_cell.find('a')
        title = title_link.text.strip() if title_link else title_cell.text.strip()
        pdf_url = title_link.get('href', '') if title_link else ''
        
        # Make URL absolute if relative
        if pdf_url and not pdf_url.startswith('http'):
            pdf_url = NGX_DOC_LIB_BASE + pdf_url
        
        # Extract date
        date_cell = cells[2] if len(cells) > 2 else None
        date_submitted = date_cell.text.strip() if date_cell else ''
        
        return {
            'company_symbol': company_symbol,
            'company_name': company_name,
            'title': title,
            'date_submitted': date_submitted,
            'pdf_url': pdf_url
        }
    
    def _parse_disclosures_alternative(self, soup, limit: int) -> List[Dict]:
        """Alternative parsing method if table not found."""
        disclosures = []
        
        # Look for disclosure links in page
        links = soup.find_all('a', href=re.compile(r'Financial_NewsDocs.*\.pdf'))
        
        for link in links[:limit]:
            href = link.get('href', '')
            title = link.text.strip()
            
            if not title or not href:
                continue
            
            # Make URL absolute
            if not href.startswith('http'):
                href = NGX_DOC_LIB_BASE + href
            
            # Try to extract company from filename or title
            company_match = re.search(r'/(\d+)_([A-Z_]+)', href)
            company_symbol = company_match.group(2).split('_')[0] if company_match else None
            
            disclosures.append({
                'company_symbol': company_symbol,
                'company_name': company_symbol,
                'title': title,
                'date_submitted': '',
                'pdf_url': href
            })
        
        return disclosures
    
    def store_disclosures(self, disclosures: List[Dict]) -> int:
        """
        Store disclosures in database.
        
        Args:
            disclosures: List of disclosure dicts
            
        Returns:
            Number of new records stored
        """
        stored = 0
        
        for disc in disclosures:
            try:
                # Check if already exists
                existing = self.db.conn.execute("""
                    SELECT id FROM corporate_disclosures WHERE pdf_url = ?
                """, [disc.get('pdf_url')]).fetchone()
                
                if existing:
                    continue
                
                self.db.conn.execute("""
                    INSERT INTO corporate_disclosures 
                    (company_symbol, company_name, title, date_submitted, pdf_url)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    disc.get('company_symbol'),
                    disc.get('company_name'),
                    disc.get('title'),
                    disc.get('date_submitted'),
                    disc.get('pdf_url')
                ])
                stored += 1
                    
            except Exception as e:
                logger.warning(f"Failed to store disclosure: {e}")
        
        self.db.conn.commit()
        logger.info(f"Stored {stored} new disclosures")
        return stored
    
    def get_disclosures(
        self, 
        limit: int = 50, 
        company: str = None,
        unprocessed_only: bool = False
    ) -> List[Dict]:
        """
        Get stored disclosures from database.
        
        Args:
            limit: Maximum records to return
            company: Filter by company symbol
            unprocessed_only: Only return disclosures without AI summary
            
        Returns:
            List of disclosure records
        """
        query = """
            SELECT id, company_symbol, company_name, title, date_submitted,
                   pdf_url, ai_summary, impact_score, impact_label, 
                   key_highlights, created_at, processed_at
            FROM corporate_disclosures
            WHERE 1=1
        """
        params = []
        
        if company:
            query += " AND company_symbol = ?"
            params.append(company.upper())
        
        if unprocessed_only:
            query += " AND ai_summary IS NULL"
        
        query += " ORDER BY date_submitted DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self.db.conn.execute(query, params).fetchall()
        
        return [
            {
                'id': row[0],
                'company_symbol': row[1],
                'company_name': row[2],
                'title': row[3],
                'date_submitted': row[4],
                'pdf_url': row[5],
                'ai_summary': row[6],
                'impact_score': row[7],
                'impact_label': row[8],
                'key_highlights': row[9],
                'created_at': row[10],
                'processed_at': row[11]
            }
            for row in rows
        ]
    
    def update_disclosure(self, disclosure_id: int, **kwargs):
        """Update a disclosure record with new data."""
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in ('pdf_text', 'ai_summary', 'impact_score', 'impact_label', 
                       'key_highlights', 'processed_at'):
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return
        
        params.append(disclosure_id)
        query = f"UPDATE corporate_disclosures SET {', '.join(updates)} WHERE id = ?"
        self.db.conn.execute(query, params)
        self.db.conn.commit()
    
    def sync(self, limit: int = 100) -> int:
        """Fetch and store latest disclosures."""
        disclosures = self.fetch_disclosures(limit)
        return self.store_disclosures(disclosures)
