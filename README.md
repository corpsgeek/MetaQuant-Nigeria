# MetaQuant Nigeria 📊

A local-first desktop stock screener for the Nigerian Stock Exchange (NGX) with fundamental filters, portfolio tracking, and AI-powered insights.

## Features

- **📈 Stock Screener** - Filter stocks by P/E ratio, dividend yield, market cap, sector, and more
- **💼 Portfolio Tracker** - Track positions, calculate P&L, and analyze performance
- **👁 Watchlist** - Monitor stocks with price targets and alerts
- **🤖 AI Insights** - Get AI-powered stock analysis using Ollama (local) or Groq (cloud)

## Tech Stack

- **GUI**: Tkinter + ttkbootstrap (dark theme)
- **Database**: DuckDB (fast columnar analytics)
- **Data Sources**: TradingView (TVDataFeed), NGX website
- **AI**: Ollama (local) + Groq (cloud fallback)

## Installation

```bash
# Clone the repository
cd MetaQuantNigeria

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install ttkbootstrap for better UI
pip install ttkbootstrap
```

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the application
python main.py
```

## Project Structure

```
MetaQuantNigeria/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── src/
│   ├── database/            # DuckDB database layer
│   ├── collectors/          # Data collectors (TradingView, NGX)
│   ├── screener/            # Screening engine with filters
│   ├── portfolio/           # Portfolio management
│   ├── ai/                  # AI insight engine
│   └── gui/                 # Tkinter GUI
│       ├── app.py           # Main application
│       ├── theme.py         # Dark theme config
│       ├── tabs/            # Screen tabs
│       └── components/      # Reusable widgets
└── data/                    # Local database storage
```

## AI Setup (Optional)

### Ollama (Local - Recommended)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```

### Groq (Cloud Fallback)
Set your API key in environment:
```bash
export GROQ_API_KEY="your-key-here"
```

## Data Sources

| Source | Data | Notes |
|--------|------|-------|
| TradingView | Historical prices, technicals | Via TVDataFeed library |
| NGX Website | Price list, corporate disclosures | 30-min delay |
| IDIA Infoware | Orderbook data | Requires login (future) |

## License

MIT License
