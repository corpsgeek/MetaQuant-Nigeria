# Configuration for MetaQuant Daemon

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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
    
    # Watchlist (default symbols to track)
    default_watchlist: list = None
    
    def __post_init__(self):
        if self.default_watchlist is None:
            self.default_watchlist = [
                'MTNN', 'DANGCEM', 'GTCO', 'ZENITHBANK', 'AIRTELAFRI',
                'BUACEMENT', 'SEPLAT', 'ACCESSCORP', 'UBA', 'FBNH',
                'NESTLE', 'BUAFOODS', 'STANBIC', 'FLOURMILL', 'WAPCO'
            ]
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Check if required config is present."""
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        if not self.telegram_chat_id:
            raise ValueError("TELEGRAM_CHAT_ID not set")
        return True
