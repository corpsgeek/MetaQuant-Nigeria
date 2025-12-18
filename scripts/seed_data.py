"""
NGX Securities Data Seeder
Populates the database with all listed securities from NGX using TradingView data.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from tvDatafeed import TvDatafeed, Interval

from src.database.db_manager import DatabaseManager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Complete list of NGX listed securities (as of 2024)
# Source: https://ngxgroup.com/exchange/data/equities-price-list/
NGX_SECURITIES = [
    # Banking
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
    {"symbol": "FBNQUEST", "name": "FBN Quest Merchant Bank Ltd", "sector": "Financial Services"},
    {"symbol": "FIRSTMONIE", "name": "FirstMonie Digital", "sector": "Financial Services"},
    
    # Insurance
    {"symbol": "AFRINSURE", "name": "African Alliance Insurance Plc", "sector": "Insurance"},
    {"symbol": "AIICO", "name": "AIICO Insurance Plc", "sector": "Insurance"},
    {"symbol": "AXAMANSARD", "name": "AXA Mansard Insurance Plc", "sector": "Insurance"},
    {"symbol": "CHIPLC", "name": "Consolidated Hallmark Insurance Plc", "sector": "Insurance"},
    {"symbol": "CORNERST", "name": "Cornerstone Insurance Plc", "sector": "Insurance"},
    {"symbol": "CUSTODIAN", "name": "Custodian Investment Plc", "sector": "Insurance"},
    {"symbol": "LASACO", "name": "LASACO Assurance Plc", "sector": "Insurance"},
    {"symbol": "LINKASSURE", "name": "Linkage Assurance Plc", "sector": "Insurance"},
    {"symbol": "MANSARD", "name": "Mansard Insurance Plc", "sector": "Insurance"},
    {"symbol": "MBENEFIT", "name": "Mutual Benefits Assurance Plc", "sector": "Insurance"},
    {"symbol": "NEM", "name": "NEM Insurance Plc", "sector": "Insurance"},
    {"symbol": "NIGERINS", "name": "Niger Insurance Plc", "sector": "Insurance"},
    {"symbol": "PRESTIGE", "name": "Prestige Assurance Plc", "sector": "Insurance"},
    {"symbol": "REGALINS", "name": "Regency Assurance Plc", "sector": "Insurance"},
    {"symbol": "ROYALEX", "name": "Royal Exchange Plc", "sector": "Insurance"},
    {"symbol": "SOVRENINS", "name": "Sovereign Trust Insurance Plc", "sector": "Insurance"},
    {"symbol": "SUNUASSUR", "name": "Sunu Assurances Nigeria Plc", "sector": "Insurance"},
    {"symbol": "VEABORIG", "name": "Veritas Kapital Assurance Plc", "sector": "Insurance"},
    
    # Consumer Goods
    {"symbol": "BUA_FOODS", "name": "BUA Foods Plc", "sector": "Consumer Goods"},
    {"symbol": "CADBURY", "name": "Cadbury Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "DANGSUGAR", "name": "Dangote Sugar Refinery Plc", "sector": "Consumer Goods"},
    {"symbol": "FLOURMILL", "name": "Flour Mills of Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "GUINNESS", "name": "Guinness Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "HONYFLOUR", "name": "Honeywell Flour Mills Plc", "sector": "Consumer Goods"},
    {"symbol": "INTBREW", "name": "International Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "MBREW", "name": "Champion Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "NASCON", "name": "NASCON Allied Industries Plc", "sector": "Consumer Goods"},
    {"symbol": "NESTLE", "name": "Nestle Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "NB", "name": "Nigerian Breweries Plc", "sector": "Consumer Goods"},
    {"symbol": "PZ", "name": "PZ Cussons Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "UNILEVER", "name": "Unilever Nigeria Plc", "sector": "Consumer Goods"},
    {"symbol": "VITAFOAM", "name": "Vitafoam Nigeria Plc", "sector": "Consumer Goods"},
    
    # Industrial Goods
    {"symbol": "AUSTINLAZ", "name": "Austin Laz & Company Plc", "sector": "Industrial Goods"},
    {"symbol": "BERGER", "name": "Berger Paints Nigeria Plc", "sector": "Industrial Goods"},
    {"symbol": "BUACEMENT", "name": "BUA Cement Plc", "sector": "Industrial Goods"},
    {"symbol": "CAPHOTEL", "name": "Capital Hotel Plc", "sector": "Industrial Goods"},
    {"symbol": "CUTIX", "name": "Cutix Plc", "sector": "Industrial Goods"},
    {"symbol": "DANGCEM", "name": "Dangote Cement Plc", "sector": "Industrial Goods"},
    {"symbol": "MEYER", "name": "Meyer Plc", "sector": "Industrial Goods"},
    {"symbol": "WAPCO", "name": "Lafarge Africa Plc", "sector": "Industrial Goods"},
    
    # Oil & Gas
    {"symbol": "ARDOVA", "name": "Ardova Plc", "sector": "Oil & Gas"},
    {"symbol": "CONOIL", "name": "Conoil Plc", "sector": "Oil & Gas"},
    {"symbol": "ETERNA", "name": "Eterna Plc", "sector": "Oil & Gas"},
    {"symbol": "JAPAULGOLD", "name": "Japaul Gold & Ventures Plc", "sector": "Oil & Gas"},
    {"symbol": "MRS", "name": "MRS Oil Nigeria Plc", "sector": "Oil & Gas"},
    {"symbol": "OANDO", "name": "Oando Plc", "sector": "Oil & Gas"},
    {"symbol": "SEPLAT", "name": "Seplat Energy Plc", "sector": "Oil & Gas"},
    {"symbol": "TOTALENERG", "name": "TotalEnergies Marketing Nigeria Plc", "sector": "Oil & Gas"},
    
    # Healthcare
    {"symbol": "AFRIPRUD", "name": "Africa Prudential Plc", "sector": "Healthcare"},
    {"symbol": "EKOCORP", "name": "Ekocorp Plc", "sector": "Healthcare"},
    {"symbol": "FIDSON", "name": "Fidson Healthcare Plc", "sector": "Healthcare"},
    {"symbol": "GLAXOSMITH", "name": "GlaxoSmithKline Consumer Nigeria Plc", "sector": "Healthcare"},
    {"symbol": "MAYBAKER", "name": "May & Baker Nigeria Plc", "sector": "Healthcare"},
    {"symbol": "MORISON", "name": "Morison Industries Plc", "sector": "Healthcare"},
    {"symbol": "NEIMETH", "name": "Neimeth International Pharmaceuticals Plc", "sector": "Healthcare"},
    {"symbol": "PHARMDEKO", "name": "Pharma-Deko Plc", "sector": "Healthcare"},
    
    # Telecommunications
    {"symbol": "AIRTELAFRI", "name": "Airtel Africa Plc", "sector": "ICT"},
    {"symbol": "MTNN", "name": "MTN Nigeria Communications Plc", "sector": "ICT"},
    {"symbol": "ETRANZACT", "name": "eTranzact International Plc", "sector": "ICT"},
    {"symbol": "CWG", "name": "CWG Plc", "sector": "ICT"},
    
    # Agriculture
    {"symbol": "ELLAHLAKES", "name": "Ellah Lakes Plc", "sector": "Agriculture"},
    {"symbol": "FTN", "name": "FTN Cocoa Processors Plc", "sector": "Agriculture"},
    {"symbol": "LIVESTOCK", "name": "Livestock Feeds Plc", "sector": "Agriculture"},
    {"symbol": "OKOMUOIL", "name": "Okomu Oil Palm Plc", "sector": "Agriculture"},
    {"symbol": "PRESCO", "name": "Presco Plc", "sector": "Agriculture"},
    
    # Construction / Real Estate
    {"symbol": "JBERGER", "name": "Julius Berger Nigeria Plc", "sector": "Construction"},
    {"symbol": "UPDC", "name": "UPDC Plc", "sector": "Real Estate"},
    {"symbol": "UPDCREIT", "name": "UPDC Real Estate Investment Trust", "sector": "Real Estate"},
    
    # Conglomerates
    {"symbol": "JOHNHOLT", "name": "John Holt Plc", "sector": "Conglomerates"},
    {"symbol": "SCOA", "name": "SCOA Nigeria Plc", "sector": "Conglomerates"},
    {"symbol": "TRANSCORP", "name": "Transnational Corporation of Nigeria Plc", "sector": "Conglomerates"},
    {"symbol": "UACN", "name": "UAC of Nigeria Plc", "sector": "Conglomerates"},
    
    # Services
    {"symbol": "ABCTRANS", "name": "ABC Transport Plc", "sector": "Services"},
    {"symbol": "CAVERTON", "name": "Caverton Offshore Support Group Plc", "sector": "Services"},
    {"symbol": "LEARNAFRCA", "name": "Learn Africa Plc", "sector": "Services"},
    {"symbol": "NAHCO", "name": "Nigerian Aviation Handling Company Plc", "sector": "Services"},
    {"symbol": "REDSTAREX", "name": "Red Star Express Plc", "sector": "Services"},
    {"symbol": "TANTALIZER", "name": "Tantalizers Plc", "sector": "Services"},
    {"symbol": "TRANSCOHOT", "name": "Transcorp Hotels Plc", "sector": "Services"},
    {"symbol": "UNIVINSURE", "name": "Universal Insurance Plc", "sector": "Services"},
]


def seed_ngx_securities(db: DatabaseManager, fetch_prices: bool = True):
    """
    Seed the database with all NGX securities.
    
    Args:
        db: Database manager instance
        fetch_prices: Whether to fetch current prices from TradingView
    """
    logger.info(f"Starting NGX securities seeding... ({len(NGX_SECURITIES)} securities)")
    
    tv = None
    if fetch_prices:
        try:
            tv = TvDatafeed()
            logger.info("TradingView connection established")
        except Exception as e:
            logger.warning(f"Could not connect to TradingView: {e}")
            fetch_prices = False
    
    success_count = 0
    error_count = 0
    
    for security in NGX_SECURITIES:
        symbol = security["symbol"]
        name = security["name"]
        sector = security["sector"]
        
        try:
            stock_data = {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "is_active": True,
                "last_updated": datetime.now(),
            }
            
            # Try to fetch current price from TradingView
            if fetch_prices and tv:
                try:
                    data = tv.get_hist(
                        symbol=symbol,
                        exchange="NGSE",  # Nigerian Stock Exchange on TradingView
                        interval=Interval.in_daily,
                        n_bars=5
                    )
                    
                    if data is not None and len(data) > 0:
                        latest = data.iloc[-1]
                        prev = data.iloc[-2] if len(data) > 1 else latest
                        
                        stock_data["last_price"] = float(latest["close"])
                        stock_data["prev_close"] = float(prev["close"])
                        stock_data["volume"] = int(latest["volume"]) if "volume" in latest else 0
                        
                        if stock_data["prev_close"] > 0:
                            change = (stock_data["last_price"] - stock_data["prev_close"]) / stock_data["prev_close"] * 100
                            stock_data["change_percent"] = round(change, 2)
                        
                        logger.info(f"✓ {symbol}: ₦{stock_data['last_price']:,.2f}")
                    else:
                        logger.warning(f"⚠ {symbol}: No price data")
                        
                except Exception as e:
                    logger.debug(f"Could not fetch price for {symbol}: {e}")
            
            # Insert/update stock in database
            db.upsert_stock(stock_data)
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to seed {symbol}: {e}")
            error_count += 1
    
    logger.info(f"\nSeeding complete! Success: {success_count}, Errors: {error_count}")
    return success_count, error_count


def seed_historical_prices(db: DatabaseManager, days: int = 365):
    """
    Seed historical price data for all stocks.
    
    Args:
        db: Database manager instance
        days: Number of days of history to fetch
    """
    logger.info(f"Fetching {days} days of historical prices...")
    
    try:
        tv = TvDatafeed()
    except Exception as e:
        logger.error(f"Could not connect to TradingView: {e}")
        return
    
    stocks = db.get_all_stocks()
    
    for stock in stocks:
        symbol = stock["symbol"]
        stock_id = stock["id"]
        
        try:
            data = tv.get_hist(
                symbol=symbol,
                exchange="NGSE",
                interval=Interval.in_daily,
                n_bars=days
            )
            
            if data is not None and len(data) > 0:
                data = data.reset_index()
                
                for _, row in data.iterrows():
                    db.insert_daily_price(
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
                
                logger.info(f"✓ {symbol}: {len(data)} days of history")
            else:
                logger.warning(f"⚠ {symbol}: No historical data")
                
        except Exception as e:
            logger.error(f"✗ {symbol}: {e}")


def main():
    """Main entry point for data seeding."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed NGX securities data")
    parser.add_argument("--no-prices", action="store_true", help="Skip fetching prices")
    parser.add_argument("--history", type=int, default=0, help="Days of historical data to fetch")
    args = parser.parse_args()
    
    db = DatabaseManager()
    db.initialize()
    
    # Seed securities
    seed_ngx_securities(db, fetch_prices=not args.no_prices)
    
    # Optionally fetch historical data
    if args.history > 0:
        seed_historical_prices(db, days=args.history)
    
    db.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
