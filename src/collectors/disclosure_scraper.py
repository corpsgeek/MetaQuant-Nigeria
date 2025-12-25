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
            except:
                pass
        
        self.db.conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_disclosures (
                id INTEGER PRIMARY KEY,
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
            response = self.session.get(NGX_DISCLOSURES_URL, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the disclosures table
            table = soup.find('table', {'id': 'company-disclosure-table'})
            if not table:
                # Try alternative selectors
                table = soup.find('table', {'class': 'dataTable'})
            
            if not table:
                logger.warning("Could not find disclosures table on page")
                # Try to parse from page content using alternative method
                return self._parse_disclosures_alternative(soup, limit)
            
            # Parse table rows
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
                self.db.conn.execute("""
                    INSERT OR IGNORE INTO corporate_disclosures 
                    (company_symbol, company_name, title, date_submitted, pdf_url)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    disc.get('company_symbol'),
                    disc.get('company_name'),
                    disc.get('title'),
                    disc.get('date_submitted'),
                    disc.get('pdf_url')
                ])
                
                if self.db.conn.total_changes > 0:
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
