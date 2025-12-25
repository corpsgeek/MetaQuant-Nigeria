# Configuration for MetaQuant Daemon

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent / '.ENV'  # Try uppercase
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Daemon configuration from environment variables."""
    
    # Telegram
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # TradingView
    tradingview_user: str = os.getenv('TRADINGVIEW_USER', '')
    tradingview_pass: str = os.getenv('TRADINGVIEW_PASS', '')
    
    # Database
    database_url: str = os.getenv('DATABASE_URL', '')
    
    # AI
    groq_api_key: str = os.getenv('GROQ_API_KEY', '')
    
    # Paths
    base_dir: Path = Path(__file__).parent
    data_dir: Path = base_dir / 'data'
    models_dir: Path = base_dir / 'models'
    
    # Alert thresholds
    pathway_bull_threshold: float = 0.60  # 60% bull probability
    flow_delta_threshold: float = 2.0     # 2 standard deviations
    ml_confidence_threshold: float = 0.75  # 75% confidence
    anomaly_severity_threshold: int = 7    # Severity out of 10
    disclosure_impact_threshold: float = 1.5  # Impact score
    
    # All securities (loaded from database)
    _all_securities: List[str] = field(default_factory=list)
    
    # Fallback watchlist if database unavailable
    default_watchlist: List[str] = field(default_factory=lambda: [
        'MTNN', 'DANGCEM', 'GTCO', 'ZENITHBANK', 'AIRTELAFRI',
        'BUACEMENT', 'SEPLAT', 'ACCESSCORP', 'UBA', 'FBNH',
        'NESTLE', 'BUAFOODS', 'STANBIC', 'FLOURMILL', 'WAPCO'
    ])
    
    def __post_init__(self):
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load all securities from database
        self._load_securities_from_db()
    
    def _load_securities_from_db(self):
        """Load all active securities from the database."""
        try:
            from src.database.db_manager import DatabaseManager
            db = DatabaseManager()
            stocks = db.get_all_stocks(active_only=True)
            
            if stocks:
                self._all_securities = [s['symbol'] for s in stocks if s.get('symbol')]
                logger.info(f"Loaded {len(self._all_securities)} securities from database")
            else:
                logger.warning("No stocks found in database, using default watchlist")
                self._all_securities = self.default_watchlist.copy()
                
        except Exception as e:
            logger.error(f"Failed to load securities from DB: {e}")
            self._all_securities = self.default_watchlist.copy()
    
    @property
    def all_securities(self) -> List[str]:
        """Get all active securities from database."""
        if not self._all_securities:
            self._load_securities_from_db()
        return self._all_securities
    
    def refresh_securities(self):
        """Force refresh securities list from database."""
        self._load_securities_from_db()
        return len(self._all_securities)
    
    def validate(self) -> bool:
        """Check if required config is present."""
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        if not self.telegram_chat_id:
            raise ValueError("TELEGRAM_CHAT_ID not set")
        return True
