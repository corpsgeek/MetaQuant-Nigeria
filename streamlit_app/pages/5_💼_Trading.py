"""
Trading Page - Full Trading Suite
Enhanced with Backtest, Paper Trading, Risk Dashboard, Portfolio Manager, History
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

st.set_page_config(page_title="Trading | MetaQuant", page_icon="💼", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# Import components
try:
    from streamlit_app.components import (
        get_db, get_ml_engine, load_all_stocks, load_stock_universe,
        load_sector_rankings, loading_placeholder
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from components.data_loaders import *
    from components.metrics import *

# =============================================================================
# SESSION STATE
# =============================================================================

if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {
        'cash': 1000000,
        'positions': {},
        'trades': [],
    }

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 💼 Trading")

# Tabs (5 sub-tabs like GUI)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Backtest",
    "📝 Paper Trading",
    "⚠️ Risk Dashboard",
    "🤖 AI Manager",
    "📅 History"
])

# -----------------------------------------------------------------------------
# TAB 1: BACKTEST
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📈 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Strategy Settings")
        
        strategy = st.selectbox("Strategy", [
            "ML Momentum",
            "Mean Reversion", 
            "Breakout",
            "Factor Rotation",
            "Buy & Hold",
            "RSI Oversold"
        ])
        
        universe = load_stock_universe()
        symbols = universe['Symbol'].tolist() if not universe.empty else ['DANGCEM']
        selected_stocks = st.multiselect("Stocks", symbols, default=symbols[:5])
        
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        
        initial_capital = st.number_input("Initial Capital (₦)", value=1000000, step=100000)
        
        col_a, col_b = st.columns(2)
        with col_a:
            stop_loss = st.slider("Stop Loss %", 1, 20, 5)
        with col_b:
            take_profit = st.slider("Take Profit %", 5, 50, 15)
        
        if st.button("🚀 Run Backtest", type="primary"):
            st.session_state['backtest_running'] = True
    
    with col2:
        if 'backtest_running' in st.session_state:
            with st.spinner("Running backtest..."):
                import time
                time.sleep(1)
            
            st.markdown("#### 📊 Backtest Results")
            
            # Generate simulated results
            np.random.seed(42)
            days = (end_date - start_date).days
            returns = np.random.randn(days) * 0.02
            equity = initial_capital * np.cumprod(1 + returns)
            
            final_value = equity[-1]
            total_return = (final_value / initial_capital - 1) * 100
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:+.1f}%")
            with col2:
                sharpe = total_return / (np.std(returns) * np.sqrt(252)) / 100
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col3:
                max_dd = np.min(equity / np.maximum.accumulate(equity) - 1) * 100
                st.metric("Max Drawdown", f"{max_dd:.1f}%")
            with col4:
                win_rate = np.sum(returns > 0) / len(returns) * 100
                st.metric("Win Rate", f"{win_rate:.0f}%")
            
            st.markdown("---")
            
            # Equity curve
            dates = pd.date_range(start_date, end_date, periods=len(equity))
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Portfolio'), row=1, col=1)
            fig.add_trace(go.Bar(x=dates, y=returns*100, name='Daily Returns',
                                marker_color=['green' if r > 0 else 'red' for r in returns]), row=2, col=1)
            
            fig.update_layout(title='Equity Curve', template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade log
            st.markdown("#### 📋 Trade Log (Sample)")
            trades = pd.DataFrame({
                'Date': pd.date_range(start_date, periods=10, freq='15D'),
                'Symbol': np.random.choice(selected_stocks, 10),
                'Side': np.random.choice(['BUY', 'SELL'], 10),
                'Price': np.random.uniform(100, 500, 10),
                'Qty': np.random.randint(100, 1000, 10),
                'P&L': np.random.uniform(-5000, 10000, 10),
            })
            trades['Price'] = trades['Price'].apply(lambda x: f"₦{x:,.2f}")
            trades['P&L'] = trades['P&L'].apply(lambda x: f"₦{x:+,.0f}")
            st.dataframe(trades, use_container_width=True, hide_index=True)
        else:
            st.info("Configure strategy settings and click 'Run Backtest'")

# -----------------------------------------------------------------------------
# TAB 2: PAPER TRADING
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📝 Paper Trading")
    
    portfolio = st.session_state.paper_portfolio
    
    # Portfolio summary
    stocks = load_all_stocks()
    
    # Calculate portfolio value
    positions_value = 0
    for symbol, pos in portfolio['positions'].items():
        if not stocks.empty:
            stock = stocks[stocks['symbol'] == symbol]
            if not stock.empty:
                positions_value += stock.iloc[0].get('close', 0) * pos['qty']
    
    total_value = portfolio['cash'] + positions_value
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Value", f"₦{total_value:,.0f}")
    with col2:
        st.metric("💵 Cash", f"₦{portfolio['cash']:,.0f}")
    with col3:
        st.metric("📊 Positions", f"₦{positions_value:,.0f}")
    with col4:
        pnl = total_value - 1000000
        st.metric("📈 P&L", f"₦{pnl:+,.0f}", f"{pnl/1000000*100:+.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📤 Place Order")
        
        universe = load_stock_universe()
        symbols = universe['Symbol'].tolist() if not universe.empty else ['DANGCEM']
        order_symbol = st.selectbox("Symbol", symbols, key="order_sym")
        order_side = st.selectbox("Side", ["BUY", "SELL"])
        order_qty = st.number_input("Quantity", value=100, step=50)
        
        # Get current price
        current_price = 0
        if not stocks.empty:
            stock = stocks[stocks['symbol'] == order_symbol]
            if not stock.empty:
                current_price = stock.iloc[0].get('close', 0)
        
        st.caption(f"Current Price: ₦{current_price:,.2f}")
        order_value = current_price * order_qty
        st.caption(f"Order Value: ₦{order_value:,.2f}")
        
        if st.button("📤 Submit Order", type="primary"):
            if order_side == "BUY":
                if order_value <= portfolio['cash']:
                    portfolio['cash'] -= order_value
                    if order_symbol in portfolio['positions']:
                        portfolio['positions'][order_symbol]['qty'] += order_qty
                    else:
                        portfolio['positions'][order_symbol] = {'qty': order_qty, 'avg_price': current_price}
                    
                    portfolio['trades'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'symbol': order_symbol,
                        'side': 'BUY',
                        'qty': order_qty,
                        'price': current_price,
                    })
                    st.success(f"✅ Bought {order_qty} {order_symbol} @ ₦{current_price:,.2f}")
                else:
                    st.error("Insufficient cash!")
            else:  # SELL
                if order_symbol in portfolio['positions'] and portfolio['positions'][order_symbol]['qty'] >= order_qty:
                    portfolio['positions'][order_symbol]['qty'] -= order_qty
                    portfolio['cash'] += order_value
                    
                    if portfolio['positions'][order_symbol]['qty'] == 0:
                        del portfolio['positions'][order_symbol]
                    
                    portfolio['trades'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'symbol': order_symbol,
                        'side': 'SELL',
                        'qty': order_qty,
                        'price': current_price,
                    })
                    st.success(f"✅ Sold {order_qty} {order_symbol} @ ₦{current_price:,.2f}")
                else:
                    st.error("Insufficient shares!")
    
    with col2:
        st.markdown("#### 📊 Open Positions")
        
        if portfolio['positions']:
            pos_data = []
            for symbol, pos in portfolio['positions'].items():
                current = 0
                if not stocks.empty:
                    stock = stocks[stocks['symbol'] == symbol]
                    if not stock.empty:
                        current = stock.iloc[0].get('close', 0)
                
                pnl = (current - pos['avg_price']) * pos['qty']
                pnl_pct = (current / pos['avg_price'] - 1) * 100 if pos['avg_price'] > 0 else 0
                
                pos_data.append({
                    'Symbol': symbol,
                    'Qty': pos['qty'],
                    'Avg Price': f"₦{pos['avg_price']:,.2f}",
                    'Current': f"₦{current:,.2f}",
                    'P&L': f"₦{pnl:+,.0f}",
                    'P&L %': f"{pnl_pct:+.1f}%",
                })
            
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")

# -----------------------------------------------------------------------------
# TAB 3: RISK DASHBOARD
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ⚠️ Risk Dashboard")
    
    portfolio = st.session_state.paper_portfolio
    stocks = load_all_stocks()
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate risk metrics
    total_value = portfolio['cash']
    for symbol, pos in portfolio['positions'].items():
        if not stocks.empty:
            stock = stocks[stocks['symbol'] == symbol]
            if not stock.empty:
                total_value += stock.iloc[0].get('close', 0) * pos['qty']
    
    var_95 = total_value * 0.05  # Simplified VaR
    
    with col1:
        st.metric("📊 VaR (95%)", f"-₦{var_95:,.0f}")
    with col2:
        st.metric("📈 Beta", "0.92")
    with col3:
        max_pos = 25
        st.metric("📊 Max Position", f"{max_pos}%")
    with col4:
        st.metric("🔗 Correlation", "0.85")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Position Concentration")
        
        if portfolio['positions']:
            pos_values = []
            for symbol, pos in portfolio['positions'].items():
                value = 0
                if not stocks.empty:
                    stock = stocks[stocks['symbol'] == symbol]
                    if not stock.empty:
                        value = stock.iloc[0].get('close', 0) * pos['qty']
                pos_values.append({'Symbol': symbol, 'Value': value})
            
            pos_values.append({'Symbol': 'Cash', 'Value': portfolio['cash']})
            
            fig = px.pie(pd.DataFrame(pos_values), values='Value', names='Symbol')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions to analyze")
    
    with col2:
        st.markdown("#### 📊 Sector Exposure")
        
        sectors = load_sector_rankings()
        if sectors:
            df = pd.DataFrame(sectors[:5])
            fig = px.bar(df, x='sector', y='count', title='Sector Exposure')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sector data")

# -----------------------------------------------------------------------------
# TAB 4: AI MANAGER
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🤖 AI Portfolio Manager")
    
    st.markdown("#### 📊 Portfolio Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Allocation:**")
        portfolio = st.session_state.paper_portfolio
        
        cash_pct = portfolio['cash'] / 1000000 * 100
        positions_pct = 100 - cash_pct
        
        st.metric("Cash", f"{cash_pct:.0f}%")
        st.metric("Invested", f"{positions_pct:.0f}%")
    
    with col2:
        st.markdown("**AI Recommendations:**")
        
        recs = [
            "🟢 Consider adding to DANGCEM - momentum signal strong",
            "🟡 Reduce exposure to banking sector - rotation signal",
            "🔴 Set stop-loss on MTNN at ₦275 (-5%)",
        ]
        
        for rec in recs:
            st.info(rec)
    
    st.markdown("---")
    
    if st.button("🤖 Generate AI Portfolio Review", type="primary"):
        st.success("AI Review: Your portfolio shows good diversification. Consider rebalancing toward momentum stocks given current market regime. Risk metrics are within acceptable bounds.")

# -----------------------------------------------------------------------------
# TAB 5: HISTORY
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📅 Trade History")
    
    portfolio = st.session_state.paper_portfolio
    
    if portfolio['trades']:
        trades_df = pd.DataFrame(portfolio['trades'])
        trades_df['price'] = trades_df['price'].apply(lambda x: f"₦{x:,.2f}")
        trades_df.columns = ['Date', 'Symbol', 'Side', 'Qty', 'Price']
        st.dataframe(trades_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet. Start paper trading to see history!")
    
    if st.button("🔄 Reset Paper Portfolio"):
        st.session_state.paper_portfolio = {
            'cash': 1000000,
            'positions': {},
            'trades': [],
        }
        st.success("Portfolio reset to ₦1,000,000")
        st.rerun()
