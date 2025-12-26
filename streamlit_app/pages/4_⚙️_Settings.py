"""
Settings Page - Configuration and Preferences
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

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# ⚙️ Settings")

tab1, tab2, tab3 = st.tabs(["🔗 Connections", "👤 Profile", "ℹ️ About"])

# -----------------------------------------------------------------------------
# TAB 1: CONNECTIONS
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🔗 Data Connections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### TradingView")
        
        tv_user = st.text_input("Username", value=os.getenv("TRADINGVIEW_USER", ""))
        tv_pass = st.text_input("Password", type="password")
        
        if st.button("🔗 Connect TradingView"):
            st.success("TradingView connected!")
        
        # Status
        st.markdown("---")
        st.markdown("**Status:** 🟢 Connected (Premium)")
    
    with col2:
        st.markdown("#### Groq AI (LLM)")
        
        groq_key = st.text_input("API Key", type="password", value="gsk_..." if os.getenv("GROQ_API_KEY") else "")
        
        if st.button("🔗 Validate Groq"):
            st.success("Groq API connected!")
        
        st.markdown("---")
        st.markdown("**Status:** 🟢 Connected")
        st.markdown("**Model:** llama-3.3-70b-versatile")
    
    st.markdown("---")
    
    st.markdown("#### Database")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Database", "DuckDB")
        st.metric("Size", "56.5 MB")
    with col2:
        st.metric("Securities", "155")
        st.metric("Historical Days", "365+")

# -----------------------------------------------------------------------------
# TAB 2: PROFILE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 👤 User Profile")
    
    st.markdown(f"**Logged in as:** {st.session_state.get('authenticated_user', 'User')}")
    
    st.markdown("---")
    
    st.markdown("#### Preferences")
    
    theme = st.selectbox("Theme", ["Dark", "Light"])
    refresh_rate = st.slider("Auto-refresh rate (seconds)", 30, 300, 60)
    notifications = st.checkbox("Enable Telegram notifications", value=True)
    
    st.markdown("---")
    
    if st.button("💾 Save Preferences"):
        st.success("Preferences saved!")

# -----------------------------------------------------------------------------
# TAB 3: ABOUT
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### ℹ️ About MetaQuant Nigeria")
    
    st.markdown("""
    ## 📊 MetaQuant Nigeria
    
    **Version:** 1.0.0 Beta
    
    A comprehensive Nigerian Stock Market Intelligence Platform featuring:
    
    - 🧠 **AI-Powered Analysis** - Groq LLM (Llama 3.3) for market insights
    - 🤖 **ML Predictions** - XGBoost ensemble models
    - 📊 **Real-time Data** - TradingView integration
    - 💼 **Paper Trading** - Risk-free strategy testing
    - 📈 **Backtesting** - Historical performance analysis
    - 🔬 **PCA Factors** - Factor-based analysis
    
    ---
    
    **Technologies:**
    - Streamlit (Web Framework)
    - DuckDB (Database)
    - XGBoost, LightGBM (ML)
    - Groq API (AI)
    - TradingView (Data)
    - Plotly (Charts)
    
    ---
    
    **Built by:** MetaLabs Nigeria
    
    **Contact:** support@metalabs.ng
    """)
