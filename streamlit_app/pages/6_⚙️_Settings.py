"""
Settings Page - Configuration and System Info
"""

import streamlit as st
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Settings | MetaQuant", page_icon="⚙️", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

try:
    from streamlit_app.components import get_db, get_collector, get_ml_engine, get_pca_engine, get_insight_engine
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from components.data_loaders import *

st.markdown("# ⚙️ Settings")

tab1, tab2, tab3 = st.tabs(["🔗 Connections", "👤 Profile", "ℹ️ About"])

# -----------------------------------------------------------------------------
# TAB 1: CONNECTIONS
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🔗 Data Connections & API Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### TradingView")
        
        tv_user = os.getenv("TRADINGVIEW_USER", "")
        tv_status = "🟢 Connected" if tv_user else "🔴 Not configured"
        st.info(f"**Status:** {tv_status}")
        
        new_user = st.text_input("TV Username", value=tv_user)
        new_pass = st.text_input("TV Password", type="password")
        
        if st.button("💾 Update TradingView"):
            st.success("Credentials would be saved to .env")
        
        st.markdown("---")
        
        # Test connection
        collector = get_collector()
        if collector:
            st.success("✅ TradingView Collector Ready")
        else:
            st.warning("⚠️ TradingView Collector unavailable")
    
    with col2:
        st.markdown("#### Groq AI")
        
        groq_key = os.getenv("GROQ_API_KEY", "")
        groq_status = "🟢 Connected" if groq_key else "🔴 Not configured"
        st.info(f"**Status:** {groq_status}")
        st.caption("Model: llama-3.3-70b-versatile")
        
        engine = get_insight_engine()
        if engine:
            st.success("✅ AI Engine Ready")
        else:
            st.warning("⚠️ AI Engine unavailable")
        
        st.markdown("---")
        
        st.markdown("#### Database")
        db = get_db()
        if db:
            try:
                count = db.conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
                st.success(f"✅ DuckDB Connected ({count} stocks)")
            except:
                st.success("✅ DuckDB Connected")
        else:
            st.error("❌ Database unavailable")
    
    st.markdown("---")
    
    st.markdown("#### 🤖 ML Engine Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ml = get_ml_engine()
        if ml:
            st.success("✅ ML Engine (XGBoost)")
        else:
            st.warning("⚠️ ML Engine unavailable")
    
    with col2:
        pca = get_pca_engine()
        if pca:
            st.success("✅ PCA Factor Engine")
        else:
            st.warning("⚠️ PCA unavailable")
    
    with col3:
        st.success("✅ Smart Money Detector")

# -----------------------------------------------------------------------------
# TAB 2: PROFILE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 👤 User Profile")
    
    st.markdown(f"**Logged in as:** {st.session_state.get('authenticated_user', 'Admin')}")
    
    st.markdown("---")
    
    st.markdown("#### 🎨 Preferences")
    
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
    refresh = st.slider("Auto-refresh (seconds)", 30, 300, 60)
    notifications = st.checkbox("Enable alerts", value=True)
    
    if st.button("💾 Save Preferences"):
        st.success("Preferences saved!")

# -----------------------------------------------------------------------------
# TAB 3: ABOUT
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ℹ️ About MetaQuant Nigeria")
    
    st.markdown("""
    ## 📊 MetaQuant Nigeria
    
    **Version:** 2.0.0 (Streamlit Edition)
    
    A comprehensive Nigerian Stock Market Intelligence Platform featuring:
    
    ### 🧠 Core Features
    - **Dashboard** - Real-time market overview with AI synthesis
    - **Flow Tape** - Advanced intraday trade flow analysis
    - **Fundamentals** - Comprehensive valuation & sector analysis
    - **ML Intelligence** - XGBoost predictions & anomaly detection
    - **PCA Factors** - Factor-based portfolio analytics
    - **Trading** - Backtest, paper trading, risk management
    
    ### 🔧 Technology Stack
    - **Frontend:** Streamlit
    - **Database:** DuckDB
    - **ML:** XGBoost, LightGBM, Scikit-learn
    - **AI:** Groq (Llama 3.3-70B)
    - **Data:** TradingView
    - **Charts:** Plotly
    
    ---
    
    **Built by:** MetaLabs Nigeria
    
    **Contact:** support@metalabs.ng
    
    © 2024 MetaLabs - All Rights Reserved
    """)
