"""
Analysis Page - Stock Screener, Universe, Watchlist, Fundamentals, Disclosures, Flow Tape
Full integration with database and real data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Analysis | MetaQuant", page_icon="📊", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_db():
    try:
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        return db
    except Exception as e:
        return None

@st.cache_resource
def get_collector():
    try:
        from src.collectors.tradingview_collector import TradingViewCollector
        return TradingViewCollector()
    except:
        return None

@st.cache_data(ttl=300)
def load_all_stocks():
    collector = get_collector()
    if collector:
        try:
            df = collector.get_all_stocks()
            return df if not df.empty else pd.DataFrame()
        except:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_universe():
    db = get_db()
    if db:
        try:
            result = db.conn.execute("""
                SELECT symbol, name, sector, subsector, market_cap
                FROM stocks
                ORDER BY symbol
            """).fetchall()
            return pd.DataFrame(result, columns=['Symbol', 'Name', 'Sector', 'Subsector', 'Market Cap'])
        except:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_disclosures():
    db = get_db()
    if db:
        try:
            result = db.conn.execute("""
                SELECT date, company_symbol, type, url
                FROM corporate_disclosures
                ORDER BY date DESC
                LIMIT 50
            """).fetchall()
            return pd.DataFrame(result, columns=['Date', 'Symbol', 'Type', 'URL'])
        except:
            pass
    return pd.DataFrame()

db = get_db()

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 📊 Stock Analysis")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Screener", 
    "📋 Universe", 
    "⭐ Watchlist",
    "💰 Fundamentals",
    "📋 Disclosures",
    "📊 Flow Tape"
])

# -----------------------------------------------------------------------------
# TAB 1: SCREENER
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Stock Screener")
    
    df = load_all_stocks()
    universe = load_universe()
    
    if not df.empty:
        # Get unique sectors from universe
        sectors = ['All'] + sorted(universe['Sector'].dropna().unique().tolist()) if not universe.empty else ['All']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sector = st.selectbox("Sector", sectors)
        with col2:
            min_price = st.number_input("Min Price", value=0.0)
        with col3:
            max_price = st.number_input("Max Price", value=10000.0)
        with col4:
            min_change = st.number_input("Min Change %", value=-100.0)
        
        if st.button("🔍 Screen", type="primary"):
            # Apply filters
            filtered = df.copy()
            
            if 'close' in filtered.columns:
                filtered = filtered[(filtered['close'] >= min_price) & (filtered['close'] <= max_price)]
            
            if 'change' in filtered.columns:
                filtered = filtered[filtered['change'] >= min_change]
            
            if sector != 'All' and not universe.empty:
                sector_symbols = universe[universe['Sector'] == sector]['Symbol'].tolist()
                filtered = filtered[filtered['symbol'].isin(sector_symbols)]
            
            st.success(f"Found {len(filtered)} stocks")
            
            if not filtered.empty:
                display = filtered[['symbol', 'close', 'change', 'volume']].copy()
                display.columns = ['Symbol', 'Price', 'Change %', 'Volume']
                st.dataframe(display, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Loading data...")

# -----------------------------------------------------------------------------
# TAB 2: UNIVERSE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📋 Stock Universe")
    
    universe = load_universe()
    
    if not universe.empty:
        search = st.text_input("🔍 Search by symbol or name")
        
        filtered = universe
        if search:
            mask = (
                filtered['Symbol'].str.contains(search.upper(), na=False) | 
                filtered['Name'].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(universe)} securities")
    else:
        st.info("Loading universe...")

# -----------------------------------------------------------------------------
# TAB 3: WATCHLIST
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⭐ Your Watchlist")
    
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["DANGCEM", "GTCO", "MTNN", "ZENITHBANK", "AIRTEL"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add symbol")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add"):
            if new_symbol.upper() and new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()}")
                st.rerun()
    
    st.markdown("---")
    
    # Get prices for watchlist
    df = load_all_stocks()
    
    for symbol in st.session_state.watchlist:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.markdown(f"**{symbol}**")
        
        with col2:
            if not df.empty and 'symbol' in df.columns:
                stock = df[df['symbol'] == symbol]
                if not stock.empty:
                    price = stock.iloc[0].get('close', 0)
                    change = stock.iloc[0].get('change', 0)
                    st.markdown(f"₦{price:,.2f} ({change:+.2f}%)")
                else:
                    st.markdown("--")
            else:
                st.markdown("--")
        
        with col3:
            pass
        
        with col4:
            if st.button("❌", key=f"rm_{symbol}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 4: FUNDAMENTALS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 💰 Fundamentals")
    
    universe = load_universe()
    symbols = universe['Symbol'].tolist() if not universe.empty else []
    
    if symbols:
        symbol = st.selectbox("Select Stock", symbols)
        
        if symbol:
            df = load_all_stocks()
            
            if not df.empty:
                stock = df[df['symbol'] == symbol]
                
                if not stock.empty:
                    s = stock.iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Key Metrics")
                        st.metric("Price", f"₦{s.get('close', 0):,.2f}")
                        st.metric("Change", f"{s.get('change', 0):+.2f}%")
                        st.metric("Volume", f"{s.get('volume', 0):,.0f}")
                        if 'market_cap_basic' in s:
                            st.metric("Market Cap", f"₦{s.get('market_cap_basic', 0)/1e9:.2f}B")
                    
                    with col2:
                        st.markdown("#### Performance")
                        if 'Perf.W' in s:
                            st.metric("1W Return", f"{s.get('Perf.W', 0):+.2f}%")
                        if 'Perf.1M' in s:
                            st.metric("1M Return", f"{s.get('Perf.1M', 0):+.2f}%")
                        if 'Perf.Y' in s:
                            st.metric("YTD Return", f"{s.get('Perf.Y', 0):+.2f}%")
                else:
                    st.info(f"No data for {symbol}")
    else:
        st.info("Loading...")

# -----------------------------------------------------------------------------
# TAB 5: DISCLOSURES
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📋 Corporate Disclosures")
    
    disclosures = load_disclosures()
    
    if not disclosures.empty:
        st.dataframe(disclosures, use_container_width=True, hide_index=True)
    else:
        st.info("No recent disclosures found")

# -----------------------------------------------------------------------------
# TAB 6: FLOW TAPE
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("### 📊 Flow Tape")
    
    df = load_all_stocks()
    
    if not df.empty and 'volume' in df.columns:
        # Calculate money flow
        df['money_flow'] = df['close'] * df['volume']
        df['direction'] = df['change'].apply(lambda x: '📈 BUY' if x > 0 else ('📉 SELL' if x < 0 else '➡️ NEUTRAL'))
        
        top_flows = df.nlargest(20, 'money_flow')[['symbol', 'direction', 'close', 'volume', 'money_flow', 'change']]
        top_flows['money_flow'] = top_flows['money_flow'].apply(lambda x: f"₦{x/1e6:.1f}M")
        top_flows.columns = ['Symbol', 'Direction', 'Price', 'Volume', 'Flow', 'Change %']
        
        st.dataframe(top_flows, use_container_width=True, hide_index=True)
    else:
        st.info("Loading flow data...")
