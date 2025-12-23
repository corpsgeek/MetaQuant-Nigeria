"""
AI Insight Engine for MetaQuant Nigeria.
Provides AI-powered stock analysis using Ollama (local) with Groq fallback.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


logger = logging.getLogger(__name__)


class InsightEngine:
    """
    AI-powered insight engine for stock analysis.
    
    Uses Ollama for local inference with Groq as a cloud fallback.
    Provides:
    - Stock analysis and recommendations
    - Portfolio health assessment
    - Market trend analysis
    - Risk assessment
    """
    
    # Default models
    OLLAMA_MODEL = "llama3.2"  # or "mistral", "phi3"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(
        self, 
        ollama_model: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None
    ):
        """
        Initialize the insight engine.
        
        Args:
            ollama_model: Ollama model to use
            groq_api_key: Groq API key for fallback
            groq_model: Groq model to use
        """
        self.ollama_model = ollama_model or self.OLLAMA_MODEL
        self.groq_model = groq_model or self.GROQ_MODEL
        self.groq_client = None
        
        logger.info(f"InsightEngine init: GROQ_AVAILABLE={GROQ_AVAILABLE}, api_key_present={bool(groq_api_key)}")
        if groq_api_key and groq_api_key.strip():
            logger.info(f"Groq API key length: {len(groq_api_key)}")
        
        if groq_api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        self._ollama_available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available and running."""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not installed")
            return False
        
        try:
            ollama.list()
            logger.info("Ollama is available")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def _generate_ollama(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response using Ollama."""
        if not self._ollama_available:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
    
    def _generate_groq(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response using Groq."""
        if not self.groq_client:
            logger.warning("Groq client is None - API key may not be set correctly")
            return None
        
        try:
            logger.info(f"Calling Groq API with model: {self.groq_model}")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            logger.info("Groq API call successful")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate AI response with fallback.
        
        Tries Ollama first, falls back to Groq if unavailable.
        """
        # Try Ollama first
        result = self._generate_ollama(prompt, system_prompt)
        if result:
            return result
        
        # Fallback to Groq
        result = self._generate_groq(prompt, system_prompt)
        if result:
            return result
        
        return "AI insights are currently unavailable. Please ensure Ollama is running or configure a Groq API key."
    
    def analyze_stock(self, stock_data: Dict[str, Any]) -> str:
        """
        Generate analysis for a single stock.
        
        Args:
            stock_data: Dictionary with stock information including:
                - symbol, name, sector
                - last_price, change_percent
                - pe_ratio, eps, dividend_yield
                - volume, market_cap
        """
        system_prompt = """You are a financial analyst specializing in Nigerian equities. 
Provide concise, actionable insights based on the data provided.
Focus on: valuation, growth potential, risks, and recommendation.
Keep response under 300 words. Use bullet points for clarity."""

        prompt = f"""Analyze this Nigerian stock:

**{stock_data.get('symbol', 'N/A')} - {stock_data.get('name', 'N/A')}**

Sector: {stock_data.get('sector', 'N/A')}
Current Price: ₦{stock_data.get('last_price', 'N/A'):,.2f}
Change: {stock_data.get('change_percent', 0):.2f}%
Volume: {stock_data.get('volume', 0):,}

Fundamentals:
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- EPS: ₦{stock_data.get('eps', 'N/A')}
- Dividend Yield: {stock_data.get('dividend_yield', 'N/A')}%
- Market Cap: ₦{stock_data.get('market_cap', 0):,.0f}
- ROE: {stock_data.get('roe', 'N/A')}%

Provide:
1. Quick take (bullish/bearish/neutral)
2. Key strengths
3. Key risks
4. Fair value estimate if possible
5. Recommendation (Buy/Hold/Sell)"""

        return self.generate(prompt, system_prompt)
    
    def analyze_portfolio(self, portfolio_summary: Dict[str, Any]) -> str:
        """
        Generate health analysis for a portfolio.
        
        Args:
            portfolio_summary: Dictionary with:
                - total_value, total_cost
                - unrealized_pnl, return_percent
                - position_count
                - sector_allocation (dict)
                - top_performers (list)
                - worst_performers (list)
        """
        system_prompt = """You are a portfolio manager reviewing a Nigerian equity portfolio.
Provide actionable recommendations for portfolio optimization.
Focus on diversification, risk management, and rebalancing opportunities.
Keep response under 400 words."""

        sector_str = "\n".join([
            f"  - {sector}: {pct:.1f}%"
            for sector, pct in portfolio_summary.get('sector_allocation', {}).items()
        ])
        
        top_str = "\n".join([
            f"  - {p.get('symbol', 'N/A')}: {p.get('return_percent', 0):.1f}%"
            for p in portfolio_summary.get('top_performers', [])[:3]
        ])
        
        worst_str = "\n".join([
            f"  - {p.get('symbol', 'N/A')}: {p.get('return_percent', 0):.1f}%"
            for p in portfolio_summary.get('worst_performers', [])[:3]
        ])

        prompt = f"""Review this portfolio:

**Portfolio Summary**
Total Value: ₦{portfolio_summary.get('total_value', 0):,.2f}
Total Cost: ₦{portfolio_summary.get('total_cost', 0):,.2f}
Unrealized P&L: ₦{portfolio_summary.get('unrealized_pnl', 0):,.2f}
Return: {portfolio_summary.get('return_percent', 0):.2f}%
Positions: {portfolio_summary.get('position_count', 0)}

**Sector Allocation**
{sector_str or '  No sector data'}

**Top Performers**
{top_str or '  No data'}

**Worst Performers**
{worst_str or '  No data'}

Provide:
1. Overall portfolio health (Excellent/Good/Fair/Poor)
2. Diversification assessment
3. Risk level (High/Medium/Low)
4. Top 3 recommendations for improvement
5. Positions to consider trimming or adding to"""

        return self.generate(prompt, system_prompt)
    
    def get_market_outlook(self, market_data: Dict[str, Any]) -> str:
        """Generate market outlook based on overall market data."""
        system_prompt = """You are a market analyst covering the Nigerian Stock Exchange.
Provide a brief market outlook based on the data.
Keep response under 200 words."""

        prompt = f"""Nigerian Stock Exchange Overview:

Total Market Cap: ₦{market_data.get('total_market_cap', 0):,.0f}
ASI (All-Share Index): {market_data.get('asi', 0):,.2f}
ASI Change: {market_data.get('asi_change', 0):.2f}%
Total Volume: {market_data.get('total_volume', 0):,}
Advancers: {market_data.get('advancers', 0)}
Decliners: {market_data.get('decliners', 0)}
Unchanged: {market_data.get('unchanged', 0)}

Provide:
1. Market sentiment (Bullish/Bearish/Neutral)
2. Key observations
3. Near-term outlook
4. Sector to watch"""

        return self.generate(prompt, system_prompt)
    
    def compare_stocks(self, stocks: List[Dict[str, Any]]) -> str:
        """Compare multiple stocks and provide recommendation."""
        system_prompt = """You are a financial analyst comparing Nigerian stocks.
Provide a clear comparison and pick a winner.
Keep response under 300 words."""

        stocks_str = "\n\n".join([
            f"""**{s.get('symbol', 'N/A')} - {s.get('name', 'N/A')}**
Price: ₦{s.get('last_price', 0):,.2f} | P/E: {s.get('pe_ratio', 'N/A')} | Div Yield: {s.get('dividend_yield', 'N/A')}%
Market Cap: ₦{s.get('market_cap', 0):,.0f} | Sector: {s.get('sector', 'N/A')}"""
            for s in stocks
        ])

        prompt = f"""Compare these Nigerian stocks:

{stocks_str}

Provide:
1. Side-by-side comparison table
2. Strengths and weaknesses of each
3. Best pick for: Value investor / Growth investor / Income investor
4. Overall winner and why"""

        return self.generate(prompt, system_prompt)
    
    def explain_indicator(self, indicator_name: str, value: float) -> str:
        """Explain a technical or fundamental indicator."""
        system_prompt = """You are a financial educator explaining technical and fundamental analysis.
Provide clear, beginner-friendly explanations.
Keep response under 150 words."""

        prompt = f"""Explain what {indicator_name} means with value of {value}:

1. What does this indicator measure?
2. Is {value} considered high, low, or normal?
3. What does this value suggest for investors?
4. Any caveats to keep in mind?"""

        return self.generate(prompt, system_prompt)
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI engine status."""
        return {
            'ollama_available': self._ollama_available,
            'ollama_model': self.ollama_model,
            'groq_available': self.groq_client is not None,
            'groq_model': self.groq_model if self.groq_client else None,
            'primary': 'ollama' if self._ollama_available else ('groq' if self.groq_client else 'none'),
        }
