"""
PDF Parser for Corporate Disclosures.
Downloads and extracts text from NGX disclosure PDFs.
"""

import logging
import requests
import tempfile
import os
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF parsing disabled. Install with: pip install pymupdf")


class PDFParser:
    """Parses PDF files and extracts text content."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize PDF parser.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs (optional)
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    @property
    def available(self) -> bool:
        """Check if PDF parsing is available."""
        return PYMUPDF_AVAILABLE
    
    def download_pdf(self, url: str) -> Optional[bytes]:
        """
        Download PDF from URL.
        
        Args:
            url: PDF URL
            
        Returns:
            PDF content as bytes or None if failed
        """
        try:
            logger.info(f"Downloading PDF: {url[:80]}...")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Verify it's a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
                logger.warning(f"URL does not return a PDF: {content_type}")
                return None
            
            return response.content
            
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            return None
    
    def extract_text(self, pdf_content: bytes, max_pages: int = 20) -> str:
        """
        Extract text from PDF content.
        
        Args:
            pdf_content: PDF file as bytes
            max_pages: Maximum pages to process
            
        Returns:
            Extracted text
        """
        if not PYMUPDF_AVAILABLE:
            logger.error("PyMuPDF not installed")
            return ""
        
        text_parts = []
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Extract text from each page
            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(doc)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return ""
    
    def extract_from_url(self, url: str, max_pages: int = 20) -> str:
        """
        Download PDF and extract text in one step.
        
        Args:
            url: PDF URL
            max_pages: Maximum pages to process
            
        Returns:
            Extracted text
        """
        pdf_content = self.download_pdf(url)
        if not pdf_content:
            return ""
        
        return self.extract_text(pdf_content, max_pages)
    
    def extract_with_cache(self, url: str, cache_key: str = None, max_pages: int = 20) -> str:
        """
        Extract text with caching support.
        
        Args:
            url: PDF URL
            cache_key: Unique key for caching (uses URL hash if not provided)
            max_pages: Maximum pages to process
            
        Returns:
            Extracted text
        """
        if not self.cache_dir:
            return self.extract_from_url(url, max_pages)
        
        # Create cache key from URL if not provided
        if not cache_key:
            import hashlib
            cache_key = hashlib.md5(url.encode()).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        # Check cache
        if cache_file.exists():
            logger.info(f"Using cached text for {cache_key}")
            return cache_file.read_text(encoding='utf-8')
        
        # Extract and cache
        text = self.extract_from_url(url, max_pages)
        
        if text:
            cache_file.write_text(text, encoding='utf-8')
            logger.info(f"Cached extracted text to {cache_file}")
        
        return text
    
    def get_pdf_info(self, pdf_content: bytes) -> dict:
        """
        Get metadata from PDF.
        
        Args:
            pdf_content: PDF file as bytes
            
        Returns:
            Dictionary with PDF metadata
        """
        if not PYMUPDF_AVAILABLE:
            return {}
        
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            info = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
            }
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return {}
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text for better AI processing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
