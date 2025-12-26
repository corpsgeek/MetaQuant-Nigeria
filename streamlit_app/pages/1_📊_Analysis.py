"""
Analysis Page - Stock Screener, Universe, Watchlist, Fundamentals
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directories for imports
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
    """Get database connection."""
    try:
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

db = get_db()

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 📊 Stock Analysis")

# Sub-navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Screener", 
    "📋 Universe", 
    "⭐ Watchlist",
    "💰 Fundamentals",
    "📊 Flow Tape"
])

# -----------------------------------------------------------------------------
# TAB 1: SCREENER
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Stock Screener")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sector = st.selectbox("Sector", ["All", "Banking", "Oil & Gas", "Consumer Goods", "Insurance", "Industrial"])
    with col2:
        min_price = st.number_input("Min Price (₦)", value=0.0)
    with col3:
        max_price = st.number_input("Max Price (₦)", value=10000.0)
    
    if st.button("🔍 Screen Stocks", type="primary"):
        if db:
            try:
                stocks = db.get_all_stocks()
                df = pd.DataFrame(stocks)
                
                # Apply filters
                if sector != "All":
                    df = df[df.get('sector', '') == sector]
                
                df = df[(df.get('close', 0) >= min_price) & (df.get('close', 0) <= max_price)]
                
                st.dataframe(df, use_container_width=True)
                st.success(f"Found {len(df)} stocks matching criteria")
            except Exception as e:
                st.error(f"Screener error: {e}")
        else:
            # Demo data
            st.info("Using demo data (database not connected)")
            demo_df = pd.DataFrame({
                'Symbol': ['DANGCEM', 'GTCO', 'MTNN', 'ZENITHBANK', 'AIRTEL'],
                'Name': ['Dangote Cement', 'GTBank', 'MTN Nigeria', 'Zenith Bank', 'Airtel Africa'],
                'Price': [485.0, 45.5, 295.0, 38.2, 2372.0],
                'Change': ['+2.1%', '-0.5%', '+1.2%', '+0.8%', '+0.4%'],
                'Volume': ['1.2M', '5.4M', '890K', '3.2M', '120K']
            })
            st.dataframe(demo_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: UNIVERSE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📋 Stock Universe")
    st.markdown("View all 155 securities on the Nigerian Stock Exchange")
    
    if db:
        try:
            stocks = db.get_all_stocks()
            df = pd.DataFrame(stocks)
            
            # Search filter
            search = st.text_input("🔍 Search by symbol or name")
            if search:
                mask = df['symbol'].str.contains(search.upper(), na=False) | \
                       df.get('name', pd.Series()).str.contains(search, case=False, na=False)
                df = df[mask]
            
            st.dataframe(df, use_container_width=True, height=500)
            st.caption(f"Showing {len(df)} securities")
        except Exception as e:
            st.error(f"Error loading universe: {e}")
    else:
        st.info("Database not connected")

# -----------------------------------------------------------------------------
# TAB 3: WATCHLIST
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⭐ Your Watchlist")
    
    # Session state for watchlist
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["DANGCEM", "GTCO", "MTNN", "ZENITHBANK", "AIRTEL"]
    
    # Add symbol
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add symbol to watchlist")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add"):
            if new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()}")
    
    # Display watchlist
    st.markdown("---")
    for i, symbol in enumerate(st.session_state.watchlist):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{symbol}**")
        with col2:
            st.markdown("₦---.--")  # Would fetch real price
        with col3:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 4: FUNDAMENTALS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 💰 Fundamentals")
    
    symbol = st.selectbox("Select Stock", ["DANGCEM", "GTCO", "MTNN", "ZENITHBANK", "AIRTEL", "NESTLE"])
    
    if symbol:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Metrics")
            st.metric("Market Cap", "₦8.5T")
            st.metric("P/E Ratio", "12.5")
            st.metric("EPS", "₦38.50")
            st.metric("Dividend Yield", "4.2%")
        
        with col2:
            st.markdown("#### Performance")
            st.metric("52W High", "₦540.00")
            st.metric("52W Low", "₦320.00")
            st.metric("YTD Return", "+18.5%")
            st.metric("Beta", "0.85")

# -----------------------------------------------------------------------------
# TAB 5: FLOW TAPE
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📊 Flow Tape")
    st.markdown("Real-time order flow analysis")
    
    # Demo flow data
    flow_data = pd.DataFrame({
        'Time': ['09:31:15', '09:31:12', '09:31:08', '09:31:05', '09:31:01'],
        'Symbol': ['DANGCEM', 'GTCO', 'MTNN', 'ZENITHBANK', 'AIRTEL'],
        'Side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
        'Price': ['₦485.00', '₦45.5', '₦295.00', '₦38.20', '₦2,372.00'],
        'Volume': ['50,000', '120,000', '25,000', '80,000', '5,000'],
        'Delta': ['+1.2M', '-5.4M', '+7.4M', '+3.0M', '-11.9M']
    })
    
    st.dataframe(flow_data, use_container_width=True)
