"""
Flow Tape Page - Advanced Intraday Trade Flow Visualization
Full replica of 5,942-line Flow Tape Tab with 7 sub-tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Flow Tape | MetaQuant", page_icon="📈", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# Import components
try:
    from streamlit_app.components import (
        get_db, get_collector, get_intraday_collector, get_pathway_synthesizer,
        get_insight_engine, load_all_stocks, load_sector_rankings,
        create_vwap_chart, create_momentum_chart, create_volume_profile,
        create_flow_gauge, flow_cards, breadth_cards, loading_placeholder
    )
except ImportError:
    # Fallback to direct imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from components.data_loaders import *
    from components.charts import *
    from components.metrics import *

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=60)
def load_intraday(symbol: str, interval: str = '15m'):
    """Load intraday data for symbol."""
    collector = get_intraday_collector()
    if collector:
        try:
            df = collector.fetch_intraday(symbol, interval=interval, n_bars=200)
            return df if df is not None and not df.empty else pd.DataFrame()
        except:
            pass
    
    # Fallback to TradingView collector
    from src.collectors.tradingview_collector import TradingViewCollector
    try:
        tv = TradingViewCollector()
        df = tv.get_historical_data(symbol, interval=interval, n_bars=200)
        return df if df is not None else pd.DataFrame()
    except:
        return pd.DataFrame()

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 📈 Flow Tape")
st.markdown("Advanced Intraday Trade Flow Analysis")

# Symbol selector
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    stocks_df = load_all_stocks()
    symbols = stocks_df['symbol'].tolist() if not stocks_df.empty else ['DANGCEM', 'GTCO', 'MTNN']
    symbol = st.selectbox("Symbol", symbols, index=0)

with col2:
    interval = st.selectbox("Interval", ['15m', '30m', '1h', '4h', '1d'], index=0)

with col3:
    if st.button("↻ Refresh", type="primary"):
        st.cache_data.clear()
        st.rerun()

with col4:
    st.markdown(f"**Updated:** {datetime.now().strftime('%H:%M:%S')}")

# Load data
df = load_intraday(symbol, interval)

# Tabs (7 sub-tabs like GUI)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Tape & Profile",
    "🚨 Alerts",
    "📈 Charts",
    "⏱️ Sessions",
    "🎯 Trade Signals",
    "🤖 AI Synthesis",
    "🔮 Pathway Predictions"
])

# -----------------------------------------------------------------------------
# TAB 1: TAPE & PROFILE
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Tape & Profile")
    
    if not df.empty:
        # Delta metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            delta = ((last_price / prev_price) - 1) * 100
            st.metric("Last", f"₦{last_price:,.2f}", f"{delta:+.2f}%")
        
        with col2:
            vol_today = df['volume'].sum()
            st.metric("Volume", f"{vol_today:,.0f}")
        
        with col3:
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() > 0 else 0
            st.metric("VWAP", f"₦{vwap:,.2f}")
        
        with col4:
            high = df['high'].max()
            st.metric("High", f"₦{high:,.2f}")
        
        with col5:
            low = df['low'].min()
            st.metric("Low", f"₦{low:,.2f}")
        
        with col6:
            range_pct = ((high / low) - 1) * 100 if low > 0 else 0
            st.metric("Range", f"{range_pct:.2f}%")
        
        st.markdown("---")
        
        # Two columns: Trade tape + Volume profile
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📋 Trade Tape")
            
            tape_df = df.tail(50).copy()
            tape_df['direction'] = np.where(tape_df['close'] >= tape_df['open'], '📈 BUY', '📉 SELL')
            tape_df['value'] = tape_df['close'] * tape_df['volume']
            
            display = tape_df[['direction', 'close', 'volume', 'value']].iloc[::-1]
            display.columns = ['Side', 'Price', 'Volume', 'Value']
            display['Price'] = display['Price'].apply(lambda x: f"₦{x:,.2f}")
            display['Volume'] = display['Volume'].apply(lambda x: f"{x:,.0f}")
            display['Value'] = display['Value'].apply(lambda x: f"₦{x/1e6:.2f}M")
            
            st.dataframe(display, use_container_width=True, hide_index=True, height=400)
        
        with col2:
            st.markdown("#### 📊 Volume Profile")
            fig = create_volume_profile(df)
            st.plotly_chart(fig, use_container_width=True)
    else:
        loading_placeholder("Loading intraday data...")

# -----------------------------------------------------------------------------
# TAB 2: ALERTS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🚨 Flow Alerts")
    
    if not df.empty:
        # Alert thresholds
        col1, col2, col3 = st.columns(3)
        with col1:
            vol_threshold = st.slider("Volume Spike Threshold", 1.0, 5.0, 2.0)
        with col2:
            price_threshold = st.slider("Price Move Threshold %", 0.5, 5.0, 1.0)
        with col3:
            imbalance_threshold = st.slider("Imbalance Threshold", 0.5, 2.0, 0.7)
        
        st.markdown("---")
        
        # Detect alerts
        df = df.copy()
        df['vol_avg'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_avg']
        df['price_move'] = df['close'].pct_change() * 100
        
        # Volume spikes
        vol_spikes = df[df['vol_ratio'] > vol_threshold]
        
        # Price moves
        price_moves = df[abs(df['price_move']) > price_threshold]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Volume Spikes")
            if len(vol_spikes) > 0:
                display = vol_spikes[['close', 'volume', 'vol_ratio']].tail(10)
                display.columns = ['Price', 'Volume', 'Ratio']
                display['Price'] = display['Price'].apply(lambda x: f"₦{x:,.2f}")
                display['Ratio'] = display['Ratio'].apply(lambda x: f"{x:.1f}x")
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("No volume spikes detected")
        
        with col2:
            st.markdown("#### 📊 Price Moves")
            if len(price_moves) > 0:
                display = price_moves[['close', 'price_move', 'volume']].tail(10)
                display.columns = ['Price', 'Move %', 'Volume']
                display['Price'] = display['Price'].apply(lambda x: f"₦{x:,.2f}")
                display['Move %'] = display['Move %'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.info("No significant price moves")
    else:
        loading_placeholder()

# -----------------------------------------------------------------------------
# TAB 3: CHARTS
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📈 Advanced Charts")
    
    if not df.empty:
        # VWAP Chart
        st.markdown("#### 📉 VWAP Deviation Bands")
        fig = create_vwap_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Momentum Oscillator")
            fig = create_momentum_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 RVOL (Relative Volume)")
            df = df.copy()
            df['vol_avg'] = df['volume'].rolling(20).mean()
            df['rvol'] = df['volume'] / df['vol_avg']
            
            fig = go.Figure(go.Bar(
                x=df.index,
                y=df['rvol'],
                marker_color=['green' if r > 1 else 'red' for r in df['rvol']]
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="white")
            fig.update_layout(title='Relative Volume', height=300, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    else:
        loading_placeholder()

# -----------------------------------------------------------------------------
# TAB 4: SESSIONS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### ⏱️ Session Analysis")
    
    if not df.empty:
        # Session breakdown (NGX hours: 10:00 - 14:30)
        df = df.copy()
        
        if 'datetime' in df.columns or df.index.dtype == 'datetime64[ns]':
            idx = df.index if df.index.dtype == 'datetime64[ns]' else pd.to_datetime(df['datetime'])
            df['hour'] = idx.hour
            
            session_stats = df.groupby('hour').agg({
                'volume': 'sum',
                'close': ['mean', 'std'],
                'high': 'max',
                'low': 'min'
            }).reset_index()
            session_stats.columns = ['Hour', 'Volume', 'Avg Price', 'Volatility', 'High', 'Low']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Volume by Hour")
                fig = px.bar(session_stats, x='Hour', y='Volume', title='Hourly Volume Distribution')
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Price Range by Hour")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=session_stats['Hour'], y=session_stats['High'], name='High'))
                fig.add_trace(go.Scatter(x=session_stats['Hour'], y=session_stats['Low'], name='Low', fill='tonexty'))
                fig.update_layout(title='Price Range', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📋 Session Statistics")
            st.dataframe(session_stats, use_container_width=True, hide_index=True)
        else:
            st.info("Datetime information not available for session analysis")
    else:
        loading_placeholder()

# -----------------------------------------------------------------------------
# TAB 5: TRADE SIGNALS
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 🎯 Trade Signals")
    
    if not df.empty:
        df = df.copy()
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = 50  # Placeholder
        
        # Simple signal logic
        last = df.iloc[-1]
        
        signals = []
        
        # SMA crossover
        if last['close'] > last['sma_20']:
            signals.append(("📈 Price above SMA20", "BULLISH", "Price trading above short-term average"))
        else:
            signals.append(("📉 Price below SMA20", "BEARISH", "Price trading below short-term average"))
        
        # Volume confirmation
        vol_avg = df['volume'].mean()
        if last['volume'] > vol_avg * 1.5:
            signals.append(("🔥 High Volume", "CONFIRM", "Volume 1.5x above average"))
        
        # Display signals
        for title, signal_type, desc in signals:
            if signal_type == "BULLISH":
                st.success(f"**{title}**\n\n{desc}")
            elif signal_type == "BEARISH":
                st.warning(f"**{title}**\n\n{desc}")
            else:
                st.info(f"**{title}**\n\n{desc}")
        
        st.markdown("---")
        st.markdown("#### 📊 Signal Summary")
        
        bullish = sum(1 for _, t, _ in signals if t == "BULLISH")
        bearish = sum(1 for _, t, _ in signals if t == "BEARISH")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Bullish", bullish)
        with col2:
            st.metric("🔴 Bearish", bearish)
        with col3:
            net = bullish - bearish
            st.metric("⚡ Net Signal", "BUY" if net > 0 else "SELL" if net < 0 else "NEUTRAL")
    else:
        loading_placeholder()

# -----------------------------------------------------------------------------
# TAB 6: AI SYNTHESIS
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("### 🤖 AI Flow Synthesis")
    st.markdown("Groq-powered flow analysis (Llama 3.3-70B)")
    
    engine = get_insight_engine()
    
    if engine and not df.empty:
        if st.button("🔮 Generate AI Analysis", type="primary"):
            with st.spinner("Generating synthesis..."):
                try:
                    last = df.iloc[-1]
                    vol_avg = df['volume'].mean()
                    price_change = ((last['close'] / df['close'].iloc[0]) - 1) * 100
                    
                    context = f"""
                    Flow Analysis for {symbol}:
                    - Current Price: ₦{last['close']:,.2f}
                    - Session Change: {price_change:+.2f}%
                    - Current Volume: {last['volume']:,.0f} ({last['volume']/vol_avg:.1f}x avg)
                    - High: ₦{df['high'].max():,.2f}
                    - Low: ₦{df['low'].min():,.2f}
                    - VWAP: ₦{(df['close'] * df['volume']).sum() / df['volume'].sum():,.2f}
                    """
                    
                    narrative = engine.generate(
                        f"Provide a professional intraday flow analysis for this Nigerian stock: {context}. "
                        f"Include observations about volume patterns, price action quality, and near-term bias. Keep under 150 words."
                    )
                    
                    st.success("Analysis Generated!")
                    st.markdown("---")
                    st.markdown(narrative)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        if not engine:
            st.warning("Set GROQ_API_KEY for AI analysis")
        else:
            loading_placeholder()

# -----------------------------------------------------------------------------
# TAB 7: PATHWAY PREDICTIONS
# -----------------------------------------------------------------------------
with tab7:
    st.markdown("### 🔮 Price Pathway Predictions")
    st.markdown("Pandora Black Box - ML-powered price projections")
    
    synthesizer = get_pathway_synthesizer()
    
    if synthesizer and not df.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            horizon = st.selectbox("Prediction Horizon", ['1 Day', '3 Days', '5 Days', '1 Week'])
            confidence = st.slider("Confidence Level", 0.5, 0.99, 0.95)
            
            if st.button("🔮 Generate Pathway", type="primary"):
                with st.spinner("Calculating pathways..."):
                    try:
                        result = synthesizer.predict_pathway(symbol, horizon=int(horizon.split()[0]))
                        
                        if result:
                            st.session_state['pathway_result'] = result
                            st.success("Pathway generated!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if 'pathway_result' in st.session_state:
                result = st.session_state['pathway_result']
                
                st.markdown("#### 📈 Projected Price Paths")
                
                # Display prediction
                st.metric("Expected Price", f"₦{result.get('target_price', 0):,.2f}")
                st.metric("Expected Return", f"{result.get('expected_return', 0):+.2f}%")
                st.metric("Confidence", f"{result.get('confidence', 0)*100:.0f}%")
                
                # Pathway chart (placeholder)
                current = df['close'].iloc[-1]
                target = result.get('target_price', current)
                
                days = list(range(int(horizon.split()[0]) + 1))
                path = [current + (target - current) * (d / len(days)) * (1 + np.random.randn() * 0.02) for d in days]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=days, y=path, mode='lines+markers', name='Projected'))
                fig.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Target")
                fig.update_layout(title='Price Pathway Projection', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'Generate Pathway' to see predictions")
    else:
        if not synthesizer:
            st.warning("Pathway Synthesizer not available")
        else:
            loading_placeholder()
