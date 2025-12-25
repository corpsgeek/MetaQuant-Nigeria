"""
AI Disclosure Analyzer for MetaQuant Nigeria.
Uses LLM to analyze corporate disclosures and generate insights.
"""

import logging
import json
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Impact score mapping
IMPACT_LABELS = {
    2: 'Very Positive',
    1: 'Positive',
    0: 'Neutral',
    -1: 'Negative',
    -2: 'Very Negative'
}


class DisclosureAnalyzer:
    """
    Analyzes corporate disclosures using AI.
    
    Generates:
    - Summary of the disclosure
    - Key highlights
    - Expected market impact
    - Affected stocks/sectors
    """
    
    def __init__(self, insight_engine=None):
        """
        Initialize the analyzer.
        
        Args:
            insight_engine: InsightEngine instance for AI generation
        """
        self.insight_engine = insight_engine
        
        # Try to import InsightEngine if not provided
        if not self.insight_engine:
            try:
                from src.ai.insight_engine import InsightEngine
                self.insight_engine = InsightEngine()
                logger.info("Disclosure Analyzer: AI Engine initialized")
            except ImportError:
                logger.warning("InsightEngine not available - AI analysis disabled")
    
    @property
    def available(self) -> bool:
        """Check if AI analysis is available."""
        return self.insight_engine is not None
    
    def analyze(self, disclosure: Dict, pdf_text: str) -> Dict:
        """
        Analyze a disclosure and generate insights.
        
        Args:
            disclosure: Disclosure metadata (company, title, date)
            pdf_text: Extracted PDF text content
            
        Returns:
            Analysis result with summary, highlights, impact, etc.
        """
        if not self.available:
            return self._fallback_analysis(disclosure, pdf_text)
        
        company = disclosure.get('company_name', 'Unknown Company')
        title = disclosure.get('title', 'Corporate Disclosure')
        
        # Truncate text if too long (keep first ~8000 chars for context)
        max_text_len = 8000
        if len(pdf_text) > max_text_len:
            pdf_text = pdf_text[:max_text_len] + "\n\n[Text truncated for analysis...]"
        
        prompt = f"""Analyze this corporate disclosure from {company}:

DISCLOSURE TITLE: {title}

CONTENT:
{pdf_text}

Please provide a structured analysis in the following JSON format:
{{
    "summary": "A 2-paragraph summary of the key points in the disclosure",
    "highlights": ["Highlight 1", "Highlight 2", "Highlight 3", "Highlight 4", "Highlight 5"],
    "impact_score": <integer from -2 to 2 where -2=Very Negative, -1=Negative, 0=Neutral, 1=Positive, 2=Very Positive>,
    "impact_reasoning": "Brief explanation of why you expect this market impact",
    "affected_stocks": ["SYMBOL1", "SYMBOL2"],
    "affected_sectors": ["Sector1", "Sector2"],
    "key_figures": {{"metric": "value"}} 
}}

Focus on:
1. Material information that could affect stock price
2. Dividends, earnings, acquisitions, management changes
3. Any regulatory or legal implications
4. Forward-looking statements
"""
        
        try:
            response = self.insight_engine.generate(
                prompt=prompt,
                context={'disclosure': disclosure},
                max_tokens=1500
            )
            
            # Parse JSON response
            result = self._parse_ai_response(response)
            result['generated_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(disclosure, pdf_text)
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response and extract structured data."""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                data = json.loads(json_match.group())
                
                # Validate and clean
                return {
                    'summary': data.get('summary', 'No summary available'),
                    'highlights': data.get('highlights', [])[:5],
                    'impact_score': max(-2, min(2, int(data.get('impact_score', 0)))),
                    'impact_label': IMPACT_LABELS.get(int(data.get('impact_score', 0)), 'Neutral'),
                    'impact_reasoning': data.get('impact_reasoning', ''),
                    'affected_stocks': data.get('affected_stocks', []),
                    'affected_sectors': data.get('affected_sectors', []),
                    'key_figures': data.get('key_figures', {}),
                    'raw_response': response
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
        
        # Fallback: use raw response as summary
        return {
            'summary': response[:1000] if response else 'Analysis failed',
            'highlights': [],
            'impact_score': 0,
            'impact_label': 'Neutral',
            'impact_reasoning': '',
            'affected_stocks': [],
            'affected_sectors': [],
            'key_figures': {},
            'raw_response': response
        }
    
    def _fallback_analysis(self, disclosure: Dict, pdf_text: str) -> Dict:
        """Generate fallback analysis when AI is not available."""
        title = disclosure.get('title', '').upper()
        
        # Simple keyword-based impact detection
        impact_score = 0
        impact_reasoning = "Automated keyword analysis"
        
        positive_keywords = ['DIVIDEND', 'PROFIT', 'GROWTH', 'ACQUISITION', 'EXPANSION', 
                            'INCREASE', 'RECORD', 'STRONG', 'BONUS']
        negative_keywords = ['LOSS', 'DECLINE', 'LAWSUIT', 'PENALTY', 'SUSPENSION',
                            'DECREASE', 'WEAK', 'RISK', 'DEFAULT']
        
        text_upper = pdf_text.upper() + ' ' + title
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_upper)
        negative_count = sum(1 for kw in negative_keywords if kw in text_upper)
        
        if positive_count > negative_count + 2:
            impact_score = 2
        elif positive_count > negative_count:
            impact_score = 1
        elif negative_count > positive_count + 2:
            impact_score = -2
        elif negative_count > positive_count:
            impact_score = -1
        
        # Extract potential highlights from title
        highlights = []
        if 'DIVIDEND' in title:
            highlights.append("Dividend-related announcement")
        if 'BOARD' in title:
            highlights.append("Board meeting or resolution")
        if 'FINANCIAL' in title or 'RESULT' in title:
            highlights.append("Financial results disclosure")
        if 'ACQUISITION' in title or 'MERGER' in title:
            highlights.append("Corporate action - M&A activity")
        if not highlights:
            highlights.append(f"Corporate disclosure: {title[:50]}")
        
        return {
            'summary': f"This is a corporate disclosure from {disclosure.get('company_name', 'the company')} "
                      f"titled '{disclosure.get('title', 'Untitled')}'. "
                      f"AI analysis is currently unavailable. Please review the original PDF for details.",
            'highlights': highlights,
            'impact_score': impact_score,
            'impact_label': IMPACT_LABELS.get(impact_score, 'Neutral'),
            'impact_reasoning': impact_reasoning,
            'affected_stocks': [disclosure.get('company_symbol')] if disclosure.get('company_symbol') else [],
            'affected_sectors': [],
            'key_figures': {},
            'generated_at': datetime.now().isoformat(),
            'is_fallback': True
        }
    
    def get_impact_badge(self, impact_score: int) -> Tuple[str, str]:
        """
        Get icon and color for impact score.
        
        Returns:
            Tuple of (icon, color)
        """
        badges = {
            2: ('🚀', '#27ae60'),   # Very Positive - green
            1: ('📈', '#2ecc71'),   # Positive - light green
            0: ('➡️', '#95a5a6'),   # Neutral - gray
            -1: ('📉', '#e67e22'),  # Negative - orange
            -2: ('⚠️', '#e74c3c'),  # Very Negative - red
        }
        return badges.get(impact_score, ('❓', '#7f8c8d'))
