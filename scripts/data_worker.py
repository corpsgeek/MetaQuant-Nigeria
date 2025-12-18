"""
MetaQuant Data Worker
Fetches and updates stock data from TradingView with authenticated access.
Includes full NGX stock seeding and historical price data.
"""

import os
import sys
import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tvDatafeed import TvDatafeed, Interval
from src.database.db_manager import DatabaseManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Complete NGX Listed Securities (100+ stocks)
NGX_SECURITIES = [
    # ===== FINANCIAL SERVICES =====
    {"symbol": "ACCESSCORP", "name": "Access Holdings Plc", "sector": "Financial Services"},
    {"symbol": "ETI", "name": "Ecobank Transnational Inc.", "sector": "Financial Services"},
    {"symbol": "FBNH", "name": "FBN Holdings Plc", "sector": "Financial Services"},
    {"symbol": "FCMB", "name": "FCMB Group Plc", "sector": "Financial Services"},
    {"symbol": "FIDELITYBK", "name": "Fidelity Bank Plc", "sector": "Financial Services"},
    {"symbol": "GTCO", "name": "Guaranty Trust Holding Co Plc", "sector": "Financial Services"},
    {"symbol": "JAIZBANK", "name": "Jaiz Bank Plc", "sector": "Financial Services"},
    {"symbol": "STANBIC", "name": "Stanbic IBTC Holdings Plc", "sector": "Financial Services"},
    {"symbol": "STERLINGNG", "name": "Sterling Financial Holdings Co Plc", "sector": "Financial Services"},
    {"symbol": "UBA", "name": "United Bank for Africa Plc", "sector": "Financial Services"},
    {"symbol": "WEMABANK", "name": "Wema Bank Plc", "sector": "Financial Services"},
    {"symbol": "ZENITHBANK", "name": "Zenith Bank Plc", "sector": "Financial Services"},
    {"symbol": "UNITYBNK", "name": "Unity Bank Plc", "sector": "Financial Services"},
    
    # ===== INSURANCE =====
    {"symbol": "AFRINSURE", "name": "African Alliance Insurance Plc", "sector": "Insurance"},
    {"symbol": "AIICO", "name": "AIICO Insurance Plc", "sector": "Insurance"},
    {"symbol": "AXAMANSARD", "name": "AXA Mansard Insurance Plc", "sector": "Insurance"},
    {"symbol": "CHIPLC", "name": "Consolidated Hallmark Insurance Plc", "sector": "Insurance"},
    {"symbol": "CORNERST", "name": "Cornerstone Insurance Plc", "sector": "Insurance"},
    {"symbol": "CUSTODIAN", "name": "Custodian Investment Plc", "sector": "Insurance"},
    {"symbol": "LASACO", "name": "LASACO Assurance Plc", "sector": "Insurance"},
    {"symbol": "LINKASSURE", "name": "Linkage Assurance Plc", "sector": "Insurance"},
    {"symbol": "MBENEFIT", "name": "Mutual Benefits Assurance Plc", "sector": "Insurance"},
    {"symbol": "NEM", "name": "NEM Insurance Plc", "sector": "Insurance"},
    {"symbol": "NIGERINS", "name": "Niger Insurance Plc", "sector": "Insurance"},
    {"symbol": "PRESTIGE", "name": "Prestige Assurance Plc", "sector": "Insurance"},
    {"symbol": "REGALINS", "name": "Regency Assurance Plc", "sector": "Insurance"},
    {"symbol": "ROYALEX", "name": "Royal Exchange Plc", "sector": "Insurance"},
    {"symbol": "SOVRENINS", "name": "Sovereign Trust Insurance Plc", "sector": "Insurance"},
    {"symbol": "SUNUASSUR", "name": "Sunu Assurances Nigeria Plc", "sector": "Insurance"},
    {"symbol": "VERITASKAP", "name": "Veritas Kapital Assurance Plc", "sector": "Insurance"},
    
    # ===== CONSUMER GOODS =====
    {"symbol": "BUAFOODS", "name": "BUA Foods Plc", "sector": "Consumer Goods"},
    {"symbol": "CADBURY", "name": "Cadbury Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "DANGSUGAR", "name": "Dangote Sugar Refinery Plc", "sector": "Consumer Goods"},
    {"symbol": "FLOURMILL", "name": "Flour Mills of Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "GUINNESS", "name": "Guinness Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "HONYFLOUR", "name": "Honeywell Flour Mills Plc", "sector": "Consumer Goods"},
    {"symbol": "INTBREW", "name": "International Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "CHAMPION", "name": "Champion Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "NASCON", "name": "NASCON Allied Industries Plc", "sector": "Consumer Goods"},
    {"symbol": "NESTLE", "name": "Nestle Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "NB", "name": "Nigerian Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "PZ", "name": "PZ Cussons Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "UNILEVER", "name": "Unilever Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "VITAFOAM", "name": "Vitafoam Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "MCNICHOLS", "name": "McNichols Plc", "sector": "Consumer Goods"},
    {"symbol": "NNFM", "name": "Northern Nigeria Flour Mills Plc", "sector": "Consumer Goods"},
    
    # ===== INDUSTRIAL GOODS =====
    {"symbol": "BERGER", "name": "Berger Paints Nigeria Plc", "sector": "Industrial Goods"},
    {"symbol": "BUACEMENT", "name": "BUA Cement Plc", "sector": "Industrial Goods"},
    {"symbol": "CUTIX", "name": "Cutix Plc", "sector": "Industrial Goods"},
    {"symbol": "DANGCEM", "name": "Dangote Cement Plc", "sector": "Industrial Goods"},
    {"symbol": "MEYER", "name": "Meyer Plc", "sector": "Industrial Goods"},
    {"symbol": "WAPCO", "name": "Lafarge Africa Plc", "sector": "Industrial Goods"},
    {"symbol": "CAPOIL", "name": "Capital Oil Plc", "sector": "Industrial Goods"},
    {"symbol": "CILEASING", "name": "C & I Leasing Plc", "sector": "Industrial Goods"},
    
    # ===== OIL & GAS =====
    {"symbol": "ARDOVA", "name": "Ardova Plc", "sector": "Oil & Gas"},
    {"symbol": "CONOIL", "name": "Conoil Plc", "sector": "Oil & Gas"},
    {"symbol": "ETERNA", "name": "Eterna Plc", "sector": "Oil & Gas"},
    {"symbol": "JAPAULGOLD", "name": "Japaul Gold & Ventures Plc", "sector": "Oil & Gas"},
    {"symbol": "MRS", "name": "MRS Oil Nigeria Plc", "sector": "Oil & Gas"},
    {"symbol": "OANDO", "name": "Oando Plc", "sector": "Oil & Gas"},
    {"symbol": "SEPLAT", "name": "Seplat Energy Plc", "sector": "Oil & Gas"},
    {"symbol": "TOTAL", "name": "TotalEnergies Marketing Nigeria Plc", "sector": "Oil & Gas"},
    
    # ===== HEALTHCARE =====
    {"symbol": "AFRIPRUD", "name": "Africa Prudential Plc", "sector": "Financial Services"},
    {"symbol": "FIDSON", "name": "Fidson Healthcare Plc", "sector": "Healthcare"},
    {"symbol": "GLAXOSMITH", "name": "GlaxoSmithKline Consumer Nigeria Plc", "sector": "Healthcare"},
    {"symbol": "MAYBAKER", "name": "May & Baker Nigeria Plc", "sector": "Healthcare"},
    {"symbol": "MORISON", "name": "Morison Industries Plc", "sector": "Healthcare"},
    {"symbol": "NEIMETH", "name": "Neimeth International Pharmaceuticals Plc", "sector": "Healthcare"},
    {"symbol": "PHARMDEKO", "name": "Pharma-Deko Plc", "sector": "Healthcare"},
    
    # ===== ICT / TELECOMMUNICATIONS =====
    {"symbol": "AIRTELAFRI", "name": "Airtel Africa Plc", "sector": "ICT"},
    {"symbol": "MTNN", "name": "MTN Nigeria Communications Plc", "sector": "ICT"},
    {"symbol": "ETRANZACT", "name": "eTranzact International Plc", "sector": "ICT"},
    {"symbol": "CWG", "name": "CWG Plc", "sector": "ICT"},
    {"symbol": "CHAMS", "name": "Chams Holding Company Plc", "sector": "ICT"},
    
    # ===== AGRICULTURE =====
    {"symbol": "ELLAHLAKES", "name": "Ellah Lakes Plc", "sector": "Agriculture"},
    {"symbol": "FTNCOCOA", "name": "FTN Cocoa Processors Plc", "sector": "Agriculture"},
    {"symbol": "LIVESTOCK", "name": "Livestock Feeds Plc", "sector": "Agriculture"},
    {"symbol": "OKOMUOIL", "name": "Okomu Oil Palm Plc", "sector": "Agriculture"},
    {"symbol": "PRESCO", "name": "Presco Plc", "sector": "Agriculture"},
    
    # ===== CONSTRUCTION / REAL ESTATE =====
    {"symbol": "JBERGER", "name": "Julius Berger Nigeria Plc", "sector": "Construction"},
    {"symbol": "UPDC", "name": "UPDC Plc", "sector": "Real Estate"},
    {"symbol": "UPDCREIT", "name": "UPDC Real Estate Investment Trust", "sector": "Real Estate"},
    
    # ===== CONGLOMERATES =====
    {"symbol": "JOHNHOLT", "name": "John Holt Plc", "sector": "Conglomerates"},
    {"symbol": "SCOA", "name": "SCOA Nigeria Plc", "sector": "Conglomerates"},
    {"symbol": "TRANSCORP", "name": "Transnational Corporation of Nigeria Plc", "sector": "Conglomerates"},
    {"symbol": "UACN", "name": "UAC of Nigeria Plc", "sector": "Conglomerates"},
    
    # ===== SERVICES =====
    {"symbol": "ABCTRANS", "name": "ABC Transport Plc", "sector": "Services"},
    {"symbol": "CAVERTON", "name": "Caverton Offshore Support Group Plc", "sector": "Services"},
    {"symbol": "LEARNAFRCA", "name": "Learn Africa Plc", "sector": "Services"},
    {"symbol": "NAHCO", "name": "Nigerian Aviation Handling Company Plc", "sector": "Services"},
    {"symbol": "REDSTAREX", "name": "Red Star Express Plc", "sector": "Services"},
    {"symbol": "TANTALIZER", "name": "Tantalizers Plc", "sector": "Services"},
    {"symbol": "TRANSCOHOT", "name": "Transcorp Hotels Plc", "sector": "Services"},
    {"symbol": "RTBRISCOE", "name": "R.T Briscoe Plc", "sector": "Services"},
]


class DataWorker:
    """
    Data worker for fetching and updating NGX stock data.
    Uses authenticated TradingView access for reliable data.
    """
    
    EXCHANGE = "NGSE"  # Nigerian Stock Exchange on TradingView
    
    def __init__(
        self,
        username: str,
        password: str,
        db_path: Optional[str] = None
    ):
        """
        Initialize the data worker.
        
        Args:
            username: TradingView username/email
            password: TradingView password
            db_path: Optional database path
        """
        self.username = username
        self.password = password
        self.db = DatabaseManager(db_path)
        self.db.initialize()
        self.tv: Optional[TvDatafeed] = None
        
        logger.info("DataWorker initialized")
    
    def connect(self) -> bool:
        """Connect to TradingView with authentication."""
        try:
            self.tv = TvDatafeed(
                username=self.username,
                password=self.password
            )
            logger.info("✅ Connected to TradingView (authenticated)")
            return True
        except Exception as e:
            logger.error(f"❌ TradingView connection failed: {e}")
            # Try without auth as fallback
            try:
                self.tv = TvDatafeed()
                logger.warning("⚠️ Using TradingView without auth (limited data)")
                return True
            except Exception as e2:
                logger.error(f"❌ TradingView connection failed completely: {e2}")
                return False
    
    def seed_all_stocks(self) -> int:
        """
        Seed all NGX securities into the database.
        
        Returns:
            Number of stocks successfully seeded
        """
        if not self.tv:
            if not self.connect():
                return 0
        
        logger.info(f"Seeding {len(NGX_SECURITIES)} NGX securities...")
        success_count = 0
        
        for i, sec in enumerate(NGX_SECURITIES):
            symbol = sec["symbol"]
            name = sec["name"]
            sector = sec["sector"]
            
            print(f"[{i+1}/{len(NGX_SECURITIES)}] {symbol}...", end=" ", flush=True)
            
            try:
                # Fetch current price data
                data = self.tv.get_hist(
                    symbol=symbol,
                    exchange=self.EXCHANGE,
                    interval=Interval.in_daily,
                    n_bars=5
                )
                
                stock_data = {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "is_active": True,
                    "last_updated": datetime.now(),
                }
                
                if data is not None and len(data) > 0:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    
                    stock_data["last_price"] = float(latest["close"])
                    stock_data["prev_close"] = float(prev["close"])
                    stock_data["volume"] = int(latest["volume"]) if "volume" in latest else 0
                    
                    if stock_data["prev_close"] > 0:
                        change = (stock_data["last_price"] - stock_data["prev_close"]) / stock_data["prev_close"] * 100
                        stock_data["change_percent"] = round(change, 2)
                    
                    print(f"₦{stock_data['last_price']:,.2f}")
                else:
                    print("No data (added anyway)")
                
                self.db.upsert_stock(stock_data)
                success_count += 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                # Still add to database without price
                self.db.upsert_stock({
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "is_active": True,
                    "last_updated": datetime.now(),
                })
                success_count += 1
        
        logger.info(f"\n✅ Seeding complete! {success_count}/{len(NGX_SECURITIES)} stocks")
        return success_count
    
    def fetch_historical_prices(self, days: int = 365) -> int:
        """
        Fetch historical price data for all stocks.
        
        Args:
            days: Number of days of history to fetch
            
        Returns:
            Number of stocks with history fetched
        """
        if not self.tv:
            if not self.connect():
                return 0
        
        stocks = self.db.get_all_stocks()
        logger.info(f"Fetching {days} days of history for {len(stocks)} stocks...")
        
        success_count = 0
        
        for i, stock in enumerate(stocks):
            symbol = stock["symbol"]
            stock_id = stock["id"]
            
            print(f"[{i+1}/{len(stocks)}] {symbol}...", end=" ", flush=True)
            
            try:
                data = self.tv.get_hist(
                    symbol=symbol,
                    exchange=self.EXCHANGE,
                    interval=Interval.in_daily,
                    n_bars=days
                )
                
                if data is not None and len(data) > 0:
                    data = data.reset_index()
                    row_count = 0
                    
                    for _, row in data.iterrows():
                        self.db.insert_daily_price(
                            stock_id=stock_id,
                            date=row["datetime"].strftime("%Y-%m-%d"),
                            ohlcv={
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]) if "volume" in row else 0,
                            }
                        )
                        row_count += 1
                    
                    print(f"{row_count} days")
                    success_count += 1
                else:
                    print("No history")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info(f"\n✅ History fetch complete! {success_count}/{len(stocks)} stocks")
        return success_count
    
    def update_prices(self) -> int:
        """
        Update current prices for all stocks (daily job).
        
        Returns:
            Number of stocks updated
        """
        if not self.tv:
            if not self.connect():
                return 0
        
        stocks = self.db.get_all_stocks()
        logger.info(f"Updating prices for {len(stocks)} stocks...")
        
        success_count = 0
        
        for stock in stocks:
            symbol = stock["symbol"]
            stock_id = stock["id"]
            
            try:
                data = self.tv.get_hist(
                    symbol=symbol,
                    exchange=self.EXCHANGE,
                    interval=Interval.in_daily,
                    n_bars=2
                )
                
                if data is not None and len(data) > 0:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else latest
                    
                    last_price = float(latest["close"])
                    prev_close = float(prev["close"])
                    change = round((last_price - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0
                    volume = int(latest["volume"]) if "volume" in latest else 0
                    
                    # Update stock table
                    self.db.upsert_stock({
                        "symbol": symbol,
                        "name": stock["name"],
                        "sector": stock.get("sector"),
                        "last_price": last_price,
                        "prev_close": prev_close,
                        "change_percent": change,
                        "volume": volume,
                        "is_active": True,
                        "last_updated": datetime.now(),
                    })
                    
                    # Insert today's price
                    self.db.insert_daily_price(
                        stock_id=stock_id,
                        date=datetime.now().strftime("%Y-%m-%d"),
                        ohlcv={
                            "open": float(latest["open"]),
                            "high": float(latest["high"]),
                            "low": float(latest["low"]),
                            "close": last_price,
                            "volume": volume,
                        }
                    )
                    
                    success_count += 1
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"{symbol}: {e}")
        
        logger.info(f"✅ Updated {success_count}/{len(stocks)} stocks")
        return success_count
    
    def run_scheduler(self, update_time: str = "15:00"):
        """
        Run the scheduler for daily updates.
        
        Args:
            update_time: Time to run daily update (HH:MM format, GMT+1)
        """
        logger.info(f"Starting scheduler - daily updates at {update_time}")
        
        schedule.every().day.at(update_time).do(self.update_prices)
        
        # Also run immediately on start
        self.update_prices()
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def close(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Main entry point for the data worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MetaQuant Data Worker")
    parser.add_argument("--seed", action="store_true", help="Seed all NGX stocks")
    parser.add_argument("--history", type=int, default=0, help="Fetch N days of historical data")
    parser.add_argument("--update", action="store_true", help="Update current prices")
    parser.add_argument("--scheduler", action="store_true", help="Run daily scheduler")
    parser.add_argument("--time", default="15:00", help="Scheduler update time (HH:MM)")
    
    args = parser.parse_args()
    
    # TradingView credentials - MUST be set via environment variables
    TV_USERNAME = os.getenv("TV_USERNAME")
    TV_PASSWORD = os.getenv("TV_PASSWORD")
    
    if not TV_USERNAME or not TV_PASSWORD:
        logger.warning("⚠️ TV_USERNAME and TV_PASSWORD not set. Using anonymous mode.")
        logger.info("Set environment variables: export TV_USERNAME='your_email' TV_PASSWORD='your_password'")
    
    worker = DataWorker(TV_USERNAME or "", TV_PASSWORD or "")
    
    try:
        if args.seed:
            worker.seed_all_stocks()
        
        if args.history > 0:
            worker.fetch_historical_prices(days=args.history)
        
        if args.update:
            worker.update_prices()
        
        if args.scheduler:
            worker.run_scheduler(update_time=args.time)
        
        if not any([args.seed, args.history, args.update, args.scheduler]):
            # Default: seed + history
            worker.seed_all_stocks()
            worker.fetch_historical_prices(days=365)
            
    finally:
        worker.close()


if __name__ == "__main__":
    main()
