"""
Trading Page - Backtest, Paper Trading, Risk Dashboard
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Trading | MetaQuant", page_icon="💼", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 💼 Trading")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Backtest", 
    "📝 Paper Trading", 
    "⚠️ Risk Dashboard",
    "📅 History"
])

# -----------------------------------------------------------------------------
# TAB 1: BACKTEST
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Strategy Settings")
        
        strategy = st.selectbox("Strategy", [
            "ML Momentum",
            "Mean Reversion",
            "Breakout",
            "Factor Rotation"
        ])
        
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
        initial_capital = st.number_input("Initial Capital (₦)", value=1000000)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                import time
                time.sleep(2)
                st.success("Backtest complete!")
    
    with col2:
        st.markdown("#### Backtest Results")
        
        # Demo results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", "+45.2%")
        with col2:
            st.metric("Sharpe Ratio", "1.85")
        with col3:
            st.metric("Max Drawdown", "-12.4%")
        with col4:
            st.metric("Win Rate", "62%")
        
        st.markdown("---")
        
        # Demo equity curve
        import plotly.graph_objects as go
        import numpy as np
        
        dates = pd.date_range('2024-01-01', periods=100)
        equity = 1000000 * (1 + np.cumsum(np.random.randn(100) * 0.01))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Portfolio Value'))
        fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Value (₦)')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: PAPER TRADING
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📝 Paper Trading")
    st.markdown("Practice trading with virtual money")
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Value", "₦1,245,000")
    with col2:
        st.metric("Cash Available", "₦345,000")
    with col3:
        st.metric("Today's P&L", "+₦12,500", "+1.02%")
    with col4:
        st.metric("Total P&L", "+₦245,000", "+24.5%")
    
    st.markdown("---")
    
    # Place order
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Place Order")
        symbol = st.text_input("Symbol", "DANGCEM")
        order_type = st.selectbox("Order Type", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", value=100)
        price = st.number_input("Price (₦)", value=485.0)
        
        if st.button("📤 Submit Order", type="primary"):
            st.success(f"Order submitted: {order_type} {quantity} {symbol} @ ₦{price}")
    
    with col2:
        st.markdown("#### Open Positions")
        
        positions_df = pd.DataFrame({
            'Symbol': ['DANGCEM', 'GTCO', 'MTNN'],
            'Qty': [100, 500, 50],
            'Avg Price': ['₦480.00', '₦44.50', '₦290.00'],
            'Current': ['₦485.00', '₦45.50', '₦295.00'],
            'P&L': ['+₦500', '+₦500', '+₦250'],
            'P&L %': ['+1.04%', '+2.25%', '+1.72%']
        })
        st.dataframe(positions_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: RISK DASHBOARD
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⚠️ Risk Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio VaR (95%)", "-₦45,000")
    with col2:
        st.metric("Beta", "0.92")
    with col3:
        st.metric("Max Position Size", "25%")
    with col4:
        st.metric("Correlation (ASI)", "0.85")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Position Concentration")
        
        concentration_df = pd.DataFrame({
            'Symbol': ['DANGCEM', 'GTCO', 'MTNN', 'CASH', 'OTHER'],
            'Value': [300000, 225000, 175000, 345000, 200000]
        })
        
        import plotly.express as px
        fig = px.pie(concentration_df, values='Value', names='Symbol', title='Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Sector Exposure")
        
        sector_df = pd.DataFrame({
            'Sector': ['Industrial', 'Banking', 'Telecom', 'Consumer', 'Cash'],
            'Exposure': [24, 18, 14, 16, 28]
        })
        
        fig = px.bar(sector_df, x='Sector', y='Exposure', title='Sector Exposure (%)')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: HISTORY
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📅 Trade History")
    
    history_df = pd.DataFrame({
        'Date': ['2024-12-26', '2024-12-25', '2024-12-24', '2024-12-23'],
        'Symbol': ['DANGCEM', 'GTCO', 'MTNN', 'DANGCEM'],
        'Type': ['BUY', 'BUY', 'SELL', 'BUY'],
        'Qty': [100, 500, 50, 100],
        'Price': ['₦480.00', '₦44.50', '₦295.00', '₦475.00'],
        'Total': ['₦48,000', '₦22,250', '₦14,750', '₦47,500'],
        'Status': ['Filled', 'Filled', 'Filled', 'Filled']
    })
    
    st.dataframe(history_df, use_container_width=True)
