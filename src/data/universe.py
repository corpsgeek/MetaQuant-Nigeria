"""
NGX Security Universe
Complete list of all listed securities on the Nigerian Stock Exchange.
Organized by sectors with metadata for easy filtering and universe creation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum


class Sector(Enum):
    """NGX Market Sectors."""
    FINANCIAL_SERVICES = "Financial Services"
    INSURANCE = "Insurance"
    CONSUMER_GOODS = "Consumer Goods"
    INDUSTRIAL_GOODS = "Industrial Goods"
    OIL_GAS = "Oil & Gas"
    HEALTHCARE = "Healthcare"
    ICT = "ICT"
    AGRICULTURE = "Agriculture"
    CONSTRUCTION = "Construction"
    REAL_ESTATE = "Real Estate"
    CONGLOMERATES = "Conglomerates"
    SERVICES = "Services"
    NATURAL_RESOURCES = "Natural Resources"


@dataclass
class Security:
    """Represents an NGX listed security."""
    symbol: str
    name: str
    sector: Sector
    subsector: Optional[str] = None
    isin: Optional[str] = None  # International Securities Identification Number
    is_active: bool = True


# Complete NGX Security Universe (Updated December 2024)
NGX_UNIVERSE: List[Security] = [
    # ========================================
    # FINANCIAL SERVICES - Banks
    # ========================================
    Security("ACCESSCORP", "Access Holdings Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("ETI", "Ecobank Transnational Incorporated", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("FBNH", "FBN Holdings Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("FCMB", "FCMB Group Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("FIDELITYBK", "Fidelity Bank Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("GTCO", "Guaranty Trust Holding Company Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("JAIZBANK", "Jaiz Bank Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("STANBIC", "Stanbic IBTC Holdings Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("STERLINGNG", "Sterling Financial Holdings Company Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("UBA", "United Bank for Africa Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("UNITYBNK", "Unity Bank Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("WEMABANK", "Wema Bank Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    Security("ZENITHBANK", "Zenith Bank Plc", Sector.FINANCIAL_SERVICES, "Banks"),
    
    # FINANCIAL SERVICES - Other Financial Institutions
    Security("AFRIPRUD", "Africa Prudential Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("AUSTINLAZ", "Austin Laz & Company Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("COURTVILLE", "Courteville Business Solutions Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("DEAPCAP", "Deap Capital Management Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("FSDH", "FSDH Holding Company Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("GEREGU", "Geregu Power Plc", Sector.FINANCIAL_SERVICES, "Power"),
    Security("ROYALEX", "Royal Exchange Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("UCAP", "United Capital Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    Security("VERITASKAP", "Veritas Kapital Assurance Plc", Sector.FINANCIAL_SERVICES, "Other Financial"),
    
    # ========================================
    # INSURANCE
    # ========================================
    Security("AFRINSURE", "African Alliance Insurance Plc", Sector.INSURANCE),
    Security("AIICO", "AIICO Insurance Plc", Sector.INSURANCE),
    Security("AXAMANSARD", "AXA Mansard Insurance Plc", Sector.INSURANCE),
    Security("CHIPLC", "Consolidated Hallmark Holdings Plc", Sector.INSURANCE),
    Security("CORNERST", "Cornerstone Insurance Plc", Sector.INSURANCE),
    Security("CUSTODIAN", "Custodian Investment Plc", Sector.INSURANCE),
    Security("GOLDINSURE", "Goldlink Insurance Plc", Sector.INSURANCE),
    Security("GUINEAINS", "Guinea Insurance Plc", Sector.INSURANCE),
    Security("LASACO", "LASACO Assurance Plc", Sector.INSURANCE),
    Security("LINKASSURE", "Linkage Assurance Plc", Sector.INSURANCE),
    Security("MANSARD", "Mansard Insurance Plc", Sector.INSURANCE),
    Security("MBENEFIT", "Mutual Benefits Assurance Plc", Sector.INSURANCE),
    Security("NEM", "NEM Insurance Plc", Sector.INSURANCE),
    Security("NIGERINS", "Niger Insurance Plc", Sector.INSURANCE),
    Security("PRESTIGE", "Prestige Assurance Plc", Sector.INSURANCE),
    Security("REGALINS", "Regency Assurance Plc", Sector.INSURANCE),
    Security("SOVRENINS", "Sovereign Trust Insurance Plc", Sector.INSURANCE),
    Security("STACO", "Staco Insurance Plc", Sector.INSURANCE),
    Security("SUNUASSUR", "Sunu Assurances Nigeria Plc", Sector.INSURANCE),
    Security("UNIVINSURE", "Universal Insurance Plc", Sector.INSURANCE),
    Security("WAPIC", "Wapic Insurance Plc", Sector.INSURANCE),
    
    # ========================================
    # CONSUMER GOODS - Food & Beverages
    # ========================================
    Security("BUAFOODS", "BUA Foods Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("CADBURY", "Cadbury Nigeria Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("DANGSUGAR", "Dangote Sugar Refinery Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("FLOURMILL", "Flour Mills of Nigeria Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("HONYFLOUR", "Honeywell Flour Mills Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("NASCON", "NASCON Allied Industries Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("NESTLE", "Nestle Nigeria Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("NNFM", "Northern Nigeria Flour Mills Plc", Sector.CONSUMER_GOODS, "Food"),
    Security("UNILEVER", "Unilever Nigeria Plc", Sector.CONSUMER_GOODS, "Personal Care"),
    Security("MCNICHOLS", "McNichols Plc", Sector.CONSUMER_GOODS, "Food"),
    
    # CONSUMER GOODS - Beverages/Breweries
    Security("CHAMPION", "Champion Breweries Plc", Sector.CONSUMER_GOODS, "Beverages"),
    Security("GUINNESS", "Guinness Nigeria Plc", Sector.CONSUMER_GOODS, "Beverages"),
    Security("INTBREW", "International Breweries Plc", Sector.CONSUMER_GOODS, "Beverages"),
    Security("NB", "Nigerian Breweries Plc", Sector.CONSUMER_GOODS, "Beverages"),
    
    # CONSUMER GOODS - Household/Personal Care
    Security("PZ", "PZ Cussons Nigeria Plc", Sector.CONSUMER_GOODS, "Personal Care"),
    Security("VITAFOAM", "Vitafoam Nigeria Plc", Sector.CONSUMER_GOODS, "Household"),
    
    # ========================================
    # INDUSTRIAL GOODS
    # ========================================
    Security("BERGER", "Berger Paints Nigeria Plc", Sector.INDUSTRIAL_GOODS, "Paints"),
    Security("BUACEMENT", "BUA Cement Plc", Sector.INDUSTRIAL_GOODS, "Cement"),
    Security("CAP", "Chemical & Allied Products Plc", Sector.INDUSTRIAL_GOODS, "Chemicals"),
    Security("CUTIX", "Cutix Plc", Sector.INDUSTRIAL_GOODS, "Cables"),
    Security("DANGCEM", "Dangote Cement Plc", Sector.INDUSTRIAL_GOODS, "Cement"),
    Security("MEYER", "Meyer Plc", Sector.INDUSTRIAL_GOODS, "Building Materials"),
    Security("NOTORE", "Notore Chemical Industries Plc", Sector.INDUSTRIAL_GOODS, "Chemicals"),
    Security("WAPCO", "Lafarge Africa Plc", Sector.INDUSTRIAL_GOODS, "Cement"),
    
    # ========================================
    # OIL & GAS
    # ========================================
    Security("ARADEL", "Aradel Holdings Plc", Sector.OIL_GAS, "Exploration & Production"),  # Recently listed
    Security("ARDOVA", "Ardova Plc", Sector.OIL_GAS, "Downstream"),
    Security("CONOIL", "Conoil Plc", Sector.OIL_GAS, "Downstream"),
    Security("ETERNA", "Eterna Plc", Sector.OIL_GAS, "Downstream"),
    Security("JAPAULGOLD", "Japaul Gold & Ventures Plc", Sector.OIL_GAS, "Services"),
    Security("MRS", "MRS Oil Nigeria Plc", Sector.OIL_GAS, "Downstream"),
    Security("OANDO", "Oando Plc", Sector.OIL_GAS, "Integrated"),
    Security("SEPLAT", "Seplat Energy Plc", Sector.OIL_GAS, "Exploration & Production"),
    Security("TOTAL", "TotalEnergies Marketing Nigeria Plc", Sector.OIL_GAS, "Downstream"),
    
    # ========================================
    # HEALTHCARE / PHARMACEUTICALS
    # ========================================
    Security("EKOCORP", "Ekocorp Plc", Sector.HEALTHCARE),
    Security("FIDSON", "Fidson Healthcare Plc", Sector.HEALTHCARE),
    Security("GLAXOSMITH", "GlaxoSmithKline Consumer Nigeria Plc", Sector.HEALTHCARE),
    Security("MAYBAKER", "May & Baker Nigeria Plc", Sector.HEALTHCARE),
    Security("MORISON", "Morison Industries Plc", Sector.HEALTHCARE),
    Security("NEIMETH", "Neimeth International Pharmaceuticals Plc", Sector.HEALTHCARE),
    Security("PHARMDEKO", "Pharma-Deko Plc", Sector.HEALTHCARE),
    
    # ========================================
    # ICT / TELECOMMUNICATIONS
    # ========================================
    Security("AIRTELAFRI", "Airtel Africa Plc", Sector.ICT, "Telecommunications"),
    Security("CHAMS", "Chams Holding Company Plc", Sector.ICT, "IT Services"),
    Security("CWG", "CWG Plc", Sector.ICT, "IT Services"),
    Security("ETRANZACT", "eTranzact International Plc", Sector.ICT, "Fintech"),
    Security("MTNN", "MTN Nigeria Communications Plc", Sector.ICT, "Telecommunications"),
    Security("NCR", "NCR Nigeria Plc", Sector.ICT, "IT Services"),
    
    # ========================================
    # AGRICULTURE
    # ========================================
    Security("ELLAHLAKES", "Ellah Lakes Plc", Sector.AGRICULTURE),
    Security("FTNCOCOA", "FTN Cocoa Processors Plc", Sector.AGRICULTURE),
    Security("LIVESTOCK", "Livestock Feeds Plc", Sector.AGRICULTURE),
    Security("OKOMUOIL", "Okomu Oil Palm Plc", Sector.AGRICULTURE),
    Security("PRESCO", "Presco Plc", Sector.AGRICULTURE),
    
    # ========================================
    # CONSTRUCTION / REAL ESTATE
    # ========================================
    Security("JBERGER", "Julius Berger Nigeria Plc", Sector.CONSTRUCTION),
    Security("UPDC", "UPDC Plc", Sector.REAL_ESTATE),
    Security("UPDCREIT", "UPDC Real Estate Investment Trust", Sector.REAL_ESTATE),
    
    # ========================================
    # CONGLOMERATES
    # ========================================
    Security("JOHNHOLT", "John Holt Plc", Sector.CONGLOMERATES),
    Security("SCOA", "SCOA Nigeria Plc", Sector.CONGLOMERATES),
    Security("TRANSCORP", "Transnational Corporation of Nigeria Plc", Sector.CONGLOMERATES),
    Security("UACN", "UAC of Nigeria Plc", Sector.CONGLOMERATES),
    
    # ========================================
    # SERVICES
    # ========================================
    Security("ABCTRANS", "ABC Transport Plc", Sector.SERVICES, "Transport"),
    Security("CAVERTON", "Caverton Offshore Support Group Plc", Sector.SERVICES, "Aviation"),
    Security("LEARNAFRCA", "Learn Africa Plc", Sector.SERVICES, "Publishing"),
    Security("NAHCO", "Nigerian Aviation Handling Company Plc", Sector.SERVICES, "Aviation"),
    Security("REDSTAREX", "Red Star Express Plc", Sector.SERVICES, "Logistics"),
    Security("RTBRISCOE", "R.T. Briscoe Plc", Sector.SERVICES, "Automobiles"),
    Security("TANTALIZER", "Tantalizers Plc", Sector.SERVICES, "Hospitality"),
    Security("TRANSCOHOT", "Transcorp Hotels Plc", Sector.SERVICES, "Hospitality"),
    
    # ========================================
    # NATURAL RESOURCES
    # ========================================
    Security("MULTIVERSE", "Multiverse Mining and Exploration Plc", Sector.NATURAL_RESOURCES),
    Security("THOMASWY", "Thomas Wyatt Nigeria Plc", Sector.NATURAL_RESOURCES),
]


class SecurityUniverse:
    """
    Manages security universes for screening and analysis.
    Allows creating custom universes based on sectors or other criteria.
    """
    
    def __init__(self):
        self._securities = {s.symbol: s for s in NGX_UNIVERSE}
        self._by_sector: Dict[Sector, List[Security]] = {}
        self._build_sector_index()
    
    def _build_sector_index(self):
        """Build sector-based index."""
        for sec in NGX_UNIVERSE:
            if sec.sector not in self._by_sector:
                self._by_sector[sec.sector] = []
            self._by_sector[sec.sector].append(sec)
    
    @property
    def all_securities(self) -> List[Security]:
        """Get all securities in the universe."""
        return NGX_UNIVERSE.copy()
    
    @property
    def all_symbols(self) -> List[str]:
        """Get all ticker symbols."""
        return list(self._securities.keys())
    
    @property
    def sectors(self) -> List[Sector]:
        """Get all available sectors."""
        return list(self._by_sector.keys())
    
    def get_security(self, symbol: str) -> Optional[Security]:
        """Get security by symbol."""
        return self._securities.get(symbol.upper())
    
    def get_by_sector(self, sector: Sector) -> List[Security]:
        """Get all securities in a sector."""
        return self._by_sector.get(sector, []).copy()
    
    def get_sector_symbols(self, sector: Sector) -> List[str]:
        """Get all symbols in a sector."""
        return [s.symbol for s in self.get_by_sector(sector)]
    
    def create_universe(self, sectors: List[Sector] = None, symbols: List[str] = None) -> List[Security]:
        """
        Create a custom universe from sectors or symbols.
        
        Args:
            sectors: List of sectors to include
            symbols: List of specific symbols to include
            
        Returns:
            List of securities matching the criteria
        """
        result = []
        
        if sectors:
            for sector in sectors:
                result.extend(self.get_by_sector(sector))
        
        if symbols:
            for sym in symbols:
                sec = self.get_security(sym)
                if sec and sec not in result:
                    result.append(sec)
        
        return result
    
    def search(self, query: str) -> List[Security]:
        """Search securities by symbol or name."""
        query = query.upper()
        return [
            s for s in NGX_UNIVERSE
            if query in s.symbol or query in s.name.upper()
        ]
    
    def get_stats(self) -> Dict[str, any]:
        """Get universe statistics."""
        return {
            "total_securities": len(NGX_UNIVERSE),
            "sectors": len(self.sectors),
            "by_sector": {
                sector.value: len(securities)
                for sector, securities in self._by_sector.items()
            }
        }
    
    def to_dict_list(self) -> List[Dict]:
        """Convert universe to list of dictionaries for database insertion."""
        return [
            {
                "symbol": s.symbol,
                "name": s.name,
                "sector": s.sector.value,
                "subsector": s.subsector,
                "is_active": s.is_active,
            }
            for s in NGX_UNIVERSE
        ]


# Pre-defined universes
class Universes:
    """Pre-defined security universes."""
    
    @staticmethod
    def ngx_all() -> SecurityUniverse:
        """All NGX listed securities."""
        return SecurityUniverse()
    
    @staticmethod
    def ngx_30() -> List[str]:
        """NGX 30 Index constituents (top 30 by market cap)."""
        return [
            "DANGCEM", "MTNN", "AIRTELAFRI", "BUACEMENT", "GTCO", 
            "ZENITHBANK", "SEPLAT", "NESTLE", "BUAFOODS", "STANBIC",
            "ACCESSCORP", "UBA", "FBNH", "FLOURMILL", "GEREGU",
            "WAPCO", "PRESCO", "OKOMUOIL", "TRANSCORP", "JBERGER",
            "GUINNESS", "NB", "TOTAL", "DANGSUGAR", "CUSTODIAN",
            "FCMB", "FIDELITYBK", "ETI", "NASCON", "ARADEL"
        ]
    
    @staticmethod
    def banks() -> List[str]:
        """Banking sector securities."""
        universe = SecurityUniverse()
        return [s.symbol for s in universe.get_by_sector(Sector.FINANCIAL_SERVICES) 
                if s.subsector == "Banks"]
    
    @staticmethod
    def oil_gas() -> List[str]:
        """Oil & Gas sector securities."""
        universe = SecurityUniverse()
        return universe.get_sector_symbols(Sector.OIL_GAS)
    
    @staticmethod
    def consumer_goods() -> List[str]:
        """Consumer Goods sector securities."""
        universe = SecurityUniverse()
        return universe.get_sector_symbols(Sector.CONSUMER_GOODS)
    
    @staticmethod
    def industrial_goods() -> List[str]:
        """Industrial Goods sector securities."""
        universe = SecurityUniverse()
        return universe.get_sector_symbols(Sector.INDUSTRIAL_GOODS)
    
    @staticmethod
    def ict() -> List[str]:
        """ICT/Telecom sector securities."""
        universe = SecurityUniverse()
        return universe.get_sector_symbols(Sector.ICT)


# Print stats when run directly
if __name__ == "__main__":
    universe = SecurityUniverse()
    stats = universe.get_stats()
    
    print(f"NGX Security Universe")
    print(f"=" * 40)
    print(f"Total Securities: {stats['total_securities']}")
    print(f"Sectors: {stats['sectors']}")
    print()
    print("Sector Breakdown:")
    for sector, count in sorted(stats['by_sector'].items(), key=lambda x: -x[1]):
        print(f"  {sector}: {count} stocks")
    
    print()
    print("Pre-defined Universes:")
    print(f"  NGX 30: {len(Universes.ngx_30())} stocks")
    print(f"  Banks: {len(Universes.banks())} stocks")
    print(f"  Oil & Gas: {len(Universes.oil_gas())} stocks")
