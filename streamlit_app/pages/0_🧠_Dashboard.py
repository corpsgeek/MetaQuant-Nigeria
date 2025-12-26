"""
Dashboard Page - Full Market Intelligence
Exact replica of the desktop GUI Market Intelligence tab
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Dashboard | MetaQuant", page_icon="🧠", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# =============================================================================
# DATABASE & DATA SOURCES
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
        return None

@st.cache_resource
def get_collector():
    """Get TradingView collector."""
    try:
        from src.collectors.tradingview_collector import TradingViewCollector
        return TradingViewCollector()
    except:
        return None

@st.cache_resource
def get_insight_engine():
    """Get AI Insight Engine."""
    try:
        from src.ai.insight_engine import InsightEngine
        api_key = os.environ.get('GROQ_API_KEY')
        if api_key:
            return InsightEngine(groq_api_key=api_key)
    except:
        pass
    return None

@st.cache_data(ttl=300)
def load_market_data():
    """Load market data from TradingView."""
    collector = get_collector()
    if collector:
        try:
            df = collector.get_all_stocks()
            if not df.empty:
                return df.to_dict('records')
        except:
            pass
    return []

@st.cache_data(ttl=300)
def load_sector_data():
    """Load sector analysis data."""
    db = get_db()
    collector = get_collector()
    
    if not db or not collector:
        return []
    
    try:
        all_stocks = collector.get_all_stocks()
        stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
        
        sector_map = {}
        try:
            results = db.conn.execute(
                "SELECT symbol, sector FROM stocks WHERE sector IS NOT NULL AND sector != ''"
            ).fetchall()
            sector_map = {row[0]: row[1] for row in results}
        except:
            pass
        
        sector_data = {}
        for s in stocks_list:
            symbol = s.get('symbol', '')
            sector = sector_map.get(symbol, 'Other')
            
            if sector not in sector_data:
                sector_data[sector] = {'stocks': [], 'gainers': 0, 'losers': 0}
            
            chg_1d = s.get('change', 0) or 0
            chg_1w = s.get('Perf.W', 0) or 0
            
            if not isinstance(chg_1d, (int, float)) or pd.isna(chg_1d):
                chg_1d = 0.0
            if not isinstance(chg_1w, (int, float)) or pd.isna(chg_1w):
                chg_1w = 0.0
            
            sector_data[sector]['stocks'].append({
                'symbol': symbol,
                'price': s.get('close', 0) or 0,
                'chg_1d': chg_1d,
                'chg_1w': chg_1w,
            })
            
            if chg_1d > 0:
                sector_data[sector]['gainers'] += 1
            elif chg_1d < 0:
                sector_data[sector]['losers'] += 1
        
        sector_rankings = []
        for sector, data in sector_data.items():
            stocks = data['stocks']
            if not stocks:
                continue
            
            avg_1d = sum(s['chg_1d'] for s in stocks) / len(stocks)
            avg_1w = sum(s['chg_1w'] for s in stocks) / len(stocks)
            
            sector_rankings.append({
                'sector': sector,
                'avg_1d': avg_1d,
                'avg_1w': avg_1w,
                'count': len(stocks),
                'gainers': data['gainers'],
                'losers': data['losers'],
            })
        
        sector_rankings.sort(key=lambda x: x['avg_1d'], reverse=True)
        return sector_rankings
    except:
        return []

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 🧠 Market Intelligence Dashboard")

# Sub-tabs (exact replica of GUI)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Market",
    "🔄 Sector Rotation", 
    "💧 Flow Monitor",
    "🕵️ Smart Money",
    "🤖 AI Synthesis"
])

stocks_data = load_market_data()
sector_data = load_sector_data()

# -----------------------------------------------------------------------------
# TAB 1: LIVE MARKET
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Live Market Overview")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("↻ Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if stocks_data:
        df = pd.DataFrame(stocks_data)
        
        total = len(df)
        gainers = len(df[df.get('change', pd.Series([0])) > 0]) if 'change' in df.columns else 0
        losers = len(df[df.get('change', pd.Series([0])) < 0]) if 'change' in df.columns else 0
        unchanged = total - gainers - losers
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Total", total)
        with col2:
            st.metric("📈 Gainers", gainers)
        with col3:
            st.metric("📉 Losers", losers)
        with col4:
            st.metric("➡️ Unchanged", unchanged)
        with col5:
            avg = df['change'].mean() if 'change' in df.columns else 0
            st.metric("📊 Avg", f"{avg:.2f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 Top Gainers")
            if 'change' in df.columns:
                top = df.nlargest(10, 'change')[['symbol', 'close', 'change', 'volume']]
                top.columns = ['Symbol', 'Price', 'Change %', 'Volume']
                st.dataframe(top, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💨 Top Losers")
            if 'change' in df.columns:
                bot = df.nsmallest(10, 'change')[['symbol', 'close', 'change', 'volume']]
                bot.columns = ['Symbol', 'Price', 'Change %', 'Volume']
                st.dataframe(bot, use_container_width=True, hide_index=True)
        
        st.markdown("#### 🔥 Most Active")
        if 'volume' in df.columns:
            active = df.nlargest(10, 'volume')[['symbol', 'close', 'change', 'volume']]
            active.columns = ['Symbol', 'Price', 'Change %', 'Volume']
            st.dataframe(active, use_container_width=True, hide_index=True)
    else:
        st.info("Loading market data...")

# -----------------------------------------------------------------------------
# TAB 2: SECTOR ROTATION
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔄 Sector Rotation Analysis")
    
    if sector_data:
        sector_df = pd.DataFrame(sector_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Sector Rankings")
            display = sector_df[['sector', 'avg_1d', 'avg_1w', 'count', 'gainers', 'losers']].copy()
            display['avg_1d'] = display['avg_1d'].apply(lambda x: f"{x:+.2f}%")
            display['avg_1w'] = display['avg_1w'].apply(lambda x: f"{x:+.2f}%")
            display.columns = ['Sector', '1D', '1W', '#', '↑', '↓']
            st.dataframe(display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📈 Performance Chart")
            fig = px.bar(sector_df, x='sector', y='avg_1d', color='avg_1d',
                        color_continuous_scale=['red', 'yellow', 'green'])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🕐 Rotation Phase")
        
        leading = sector_df.head(3)['sector'].tolist()
        lagging = sector_df.tail(3)['sector'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Leading:** {', '.join(leading)}")
        with col2:
            st.error(f"**Lagging:** {', '.join(lagging)}")
    else:
        st.info("Loading sector data...")

# -----------------------------------------------------------------------------
# TAB 3: FLOW MONITOR
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 💧 Flow Monitor")
    
    if stocks_data:
        df = pd.DataFrame(stocks_data)
        
        if 'Perf.W' in df.columns:
            perf = df['Perf.W'].fillna(0)
            
            strong_in = len(df[perf > 5])
            mod_in = len(df[(perf > 0) & (perf <= 5)])
            neutral = len(df[perf == 0])
            mod_out = len(df[(perf < 0) & (perf >= -5)])
            strong_out = len(df[perf < -5])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🚀 Strong In", strong_in)
            with col2:
                st.metric("📈 Inflow", mod_in)
            with col3:
                st.metric("➡️ Neutral", neutral)
            with col4:
                st.metric("📉 Outflow", mod_out)
            with col5:
                st.metric("💨 Strong Out", strong_out)
            
            st.markdown("---")
            
            buying = (strong_in + mod_in) / len(df) * 100
            selling = (strong_out + mod_out) / len(df) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Buying", f"{buying:.1f}%")
            with col2:
                st.metric("📉 Selling", f"{selling:.1f}%")
            with col3:
                st.metric("⚡ Net", f"{buying - selling:+.1f}%")
            
            st.progress(buying / 100)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚀 Top Inflows")
                top = df.nlargest(10, 'Perf.W')[['symbol', 'change', 'Perf.W']]
                top.columns = ['Symbol', '1D %', '1W %']
                st.dataframe(top, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 💨 Top Outflows")
                bot = df.nsmallest(10, 'Perf.W')[['symbol', 'change', 'Perf.W']]
                bot.columns = ['Symbol', '1D %', '1W %']
                st.dataframe(bot, use_container_width=True, hide_index=True)
    else:
        st.info("Loading flow data...")

# -----------------------------------------------------------------------------
# TAB 4: SMART MONEY
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🕵️ Smart Money Detector")
    
    if stocks_data:
        df = pd.DataFrame(stocks_data)
        
        if 'volume' in df.columns and 'change' in df.columns:
            df['vol_avg'] = df['volume'].mean()
            df['vol_ratio'] = df['volume'] / df['vol_avg']
            
            unusual = df[df['vol_ratio'] > 2].sort_values('vol_ratio', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔥 Unusual Volume")
                if len(unusual) > 0:
                    disp = unusual[['symbol', 'close', 'change', 'vol_ratio']].head(10)
                    disp['vol_ratio'] = disp['vol_ratio'].apply(lambda x: f"{x:.1f}x")
                    disp.columns = ['Symbol', 'Price', 'Change', 'Vol Ratio']
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No unusual volume")
            
            with col2:
                st.markdown("#### 📊 Market Regime")
                
                gainers = len(df[df['change'] > 0])
                losers = len(df[df['change'] < 0])
                
                if gainers > losers * 1.5:
                    st.success("🟢 RISK-ON (Bullish)")
                elif losers > gainers * 1.5:
                    st.error("🔴 RISK-OFF (Bearish)")
                else:
                    st.warning("🟡 NEUTRAL")
                
                st.metric("A/D Ratio", f"{gainers/losers:.2f}" if losers else "∞")
            
            st.markdown("---")
            st.markdown("#### 💹 Breakout Candidates")
            
            breakouts = df[(df['change'] > 3) & (df['volume'] > df['volume'].mean() * 1.5)]
            if len(breakouts) > 0:
                disp = breakouts[['symbol', 'close', 'change', 'volume']].head(10)
                disp.columns = ['Symbol', 'Price', 'Change', 'Volume']
                st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info("No breakout candidates today")
    else:
        st.info("Loading data...")

# -----------------------------------------------------------------------------
# TAB 5: AI SYNTHESIS
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 🤖 AI Market Synthesis")
    st.markdown("Groq-powered (Llama 3.3-70B)")
    
    engine = get_insight_engine()
    
    if engine:
        if st.button("🔮 Generate AI Synthesis", type="primary"):
            with st.spinner("Generating..."):
                try:
                    if stocks_data:
                        df = pd.DataFrame(stocks_data)
                        gainers = len(df[df.get('change', pd.Series([0])) > 0]) if 'change' in df.columns else 0
                        losers = len(df[df.get('change', pd.Series([0])) < 0]) if 'change' in df.columns else 0
                        total = len(df)
                        
                        top_gainers = df.nlargest(5, 'change')['symbol'].tolist() if 'change' in df.columns else []
                        top_losers = df.nsmallest(5, 'change')['symbol'].tolist() if 'change' in df.columns else []
                        
                        context = f"""
                        NGX Summary:
                        - Total: {total}, Gainers: {gainers}, Losers: {losers}
                        - Top gainers: {', '.join(top_gainers)}
                        - Top losers: {', '.join(top_losers)}
                        - Leading sectors: {', '.join([s['sector'] for s in sector_data[:3]]) if sector_data else 'N/A'}
                        """
                        
                        narrative = engine.generate(
                            f"Brief market commentary for NGX: {context}. Keep under 150 words."
                        )
                        
                        st.success("Generated!")
                        st.markdown("---")
                        st.markdown(narrative)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("#### 📝 Quick Insights")
        
        if stocks_data and sector_data:
            col1, col2 = st.columns(2)
            
            with col1:
                df = pd.DataFrame(stocks_data)
                if 'change' in df.columns:
                    g = len(df[df['change'] > 0])
                    l = len(df[df['change'] < 0])
                    if g > l:
                        st.success(f"✅ Bullish ({g}↑ vs {l}↓)")
                    else:
                        st.error(f"⚠️ Bearish ({g}↑ vs {l}↓)")
            
            with col2:
                if sector_data:
                    leader = sector_data[0]
                    st.info(f"🏆 {leader['sector']} ({leader['avg_1d']:+.2f}%)")
    else:
        st.warning("Set GROQ_API_KEY for AI")
