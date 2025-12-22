# MetaQuant Nigeria 📊

A local-first desktop stock screener for the Nigerian Stock Exchange (NGX) with real-time market data, microstructure analysis, portfolio tracking, and AI-powered insights.

## ✨ Features

### 🔴 Live Market View (NEW)
- **Real-time prices** from TradingView (143 NGX stocks)
- **Sector Heatmap** - Click to drill-down into any sector
- **Market Breadth** - Advancers vs decliners visual bar
- **Top Movers** - Dynamic gainers and losers
- **Volume Leaders** - Stocks with unusual activity
- **Auto-refresh** every 60 seconds

### 📅 History Tab
- View historical market data by date (Dec 2023 - Present)
- Accurate day-over-day change calculations
- Top gainers/losers for any historical date
- Volume and performance metrics

### 📈 Stock Screener
- Filter by P/E ratio, dividend yield, market cap, sector
- Sortable columns with visual indicators
- Click any stock for detailed technicals

### 💼 Portfolio & Watchlist
- Track positions and calculate P&L
- Set price targets and alerts
- Performance analytics

### 🤖 AI Insights
- Stock analysis using Ollama (local) or Groq (cloud)
- Technical indicator interpretation
- Buy/Sell/Hold recommendations

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **GUI** | Tkinter + ttkbootstrap (dark theme) |
| **Database** | DuckDB (columnar analytics) |
| **Real-time Data** | `tradingview-screener` |
| **Historical Data** | `tvdatafeed` (Python 3.12) |
| **Technical Analysis** | `tradingview-ta` |
| **AI** | Ollama / Groq API |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MetaQuantNigeria.git
cd MetaQuantNigeria

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TradingView packages
pip install tradingview-screener tradingview-ta

# Run the application
python main.py
```

---

## 📊 Data Collection

### How It Works
1. **Real-time Data**: `tradingview-screener` fetches all 143 NGX stocks with current prices, volume, and change %
2. **Historical Backfill**: `tvdatafeed` pulls 2+ years of OHLCV data (requires Python 3.12)
3. **Automated Updates**: macOS launchd scheduler runs daily at 3 PM after market close

### Daily Data Loader
```bash
# View market snapshot
python scripts/load_market_data.py --snapshot

# Load today's data into database
python scripts/load_market_data.py
```

### Historical Backfill (Python 3.12)
```bash
# Create Python 3.12 environment (one-time)
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/pip install git+https://github.com/rongardF/tvdatafeed.git pandas duckdb

# Backfill 2 years of data
.venv312/bin/python scripts/backfill_historical.py --days 730
```

### Automatic Scheduling (macOS)
```bash
# Enable daily collection at 3 PM
cp scripts/com.metaquant.ngx-data-loader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.metaquant.ngx-data-loader.plist

# Check status
launchctl list | grep metaquant

# Disable
launchctl unload ~/Library/LaunchAgents/com.metaquant.ngx-data-loader.plist
```

---

## 📁 Project Structure

```
MetaQuantNigeria/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── src/
│   ├── analysis/
│   │   └── microstructure.py   # RVOL, momentum, breadth
│   ├── collectors/
│   │   └── tradingview_collector.py
│   ├── database/
│   │   └── db_manager.py       # DuckDB operations
│   ├── gui/
│   │   ├── app.py              # Main window
│   │   ├── tabs/
│   │   │   ├── live_market_tab.py   # Real-time view
│   │   │   ├── history_tab.py       # Historical data
│   │   │   ├── screener_tab.py      # Stock filtering
│   │   │   └── ...
│   │   └── components/
│   │       ├── stock_detail_dialog.py   # Stock popup
│   │       └── sector_detail_dialog.py  # Sector popup
│   └── ai/                     # AI insights
├── scripts/
│   ├── load_market_data.py     # Daily data loader
│   ├── backfill_historical.py  # Historical fetch
│   ├── data_scheduler.py       # Scheduler
│   └── com.metaquant.ngx-data-loader.plist  # launchd config
└── data/
    └── metaquant.db            # DuckDB database
```

---

## 🔧 Implementation Notes

### Data Sources Approach
- **Primary source**: TradingView (`tradingview-screener`) - provides reliable real-time data for all 143 NGX-listed stocks
- **Exchange identifier**: `NSENG` (not `NGSE`)
- **Screener**: `nigeria`

### Change % Calculation
- **Today's data**: Uses TradingView's actual change % (stored in `change_pct` column)
- **Historical data**: Calculated day-over-day from close prices

### Market Hours Detection
- NGX trading hours: Mon-Fri, 10:00 AM - 2:30 PM WAT (GMT+1)
- Live Market tab shows open/closed status with auto-detection

---

## 🤖 AI Setup (Optional)

### Ollama (Local - Recommended)
```bash
# Install from https://ollama.ai
ollama pull llama3.2
```

### Groq (Cloud)
```bash
export GROQ_API_KEY="your-key-here"
```

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [TradingView](https://tradingview.com) for market data
- [DuckDB](https://duckdb.org) for blazing-fast analytics
- Nigerian Stock Exchange for the market we love 🇳🇬
