"""
MetaQuant Nigeria - Streamlit Web App
Main entry point with authentication
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config must be first Streamlit command
st.set_page_config(
    page_title="MetaQuant Nigeria",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# AUTHENTICATION
# =============================================================================

def check_password():
    """Returns `True` if the user has entered a correct password."""
    
    # Get credentials from environment or use defaults
    USERS = {
        os.getenv("STREAMLIT_USER", "admin"): os.getenv("STREAMLIT_PASSWORD", "metaquant2024"),
    }
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in USERS:
            if st.session_state["password"] == USERS[st.session_state["username"]]:
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = st.session_state["username"]
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # First run or logged out
    if "password_correct" not in st.session_state:
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("# 📊 MetaQuant Nigeria")
            st.markdown("### Nigerian Stock Market Intelligence")
            st.markdown("---")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", use_container_width=True)
        return False
    
    # Password incorrect
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("# 📊 MetaQuant Nigeria")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Login", on_click=password_entered, type="primary", use_container_width=True)
            st.error("😕 Invalid username or password")
        return False
    
    return True


# =============================================================================
# MAIN APP
# =============================================================================

if check_password():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.get('authenticated_user', 'User')}")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["password_correct"] = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 MetaQuant Nigeria")
        st.markdown("v1.0.0 Beta")
        st.markdown("""
        **Features:**
        - 🧠 Market Intelligence
        - 📈 Stock Analysis
        - 🤖 ML Predictions
        - 💼 Trading Tools
        """)
    
    # Main content - Quick overview
    st.markdown("# 📊 MetaQuant Nigeria")
    st.markdown("### Nigerian Stock Market Intelligence Platform")
    
    # Quick navigation with icons
    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🧠 Dashboard")
        st.markdown("Live market, sector rotation, flow analysis, smart money, AI synthesis")
        st.page_link("pages/0_🧠_Dashboard.py", label="Open Dashboard →")
    
    with col2:
        st.markdown("#### 📊 Analysis")
        st.markdown("Screener, universe, watchlist, fundamentals, disclosures")
        st.page_link("pages/1_📊_Analysis.py", label="Open Analysis →")
    
    with col3:
        st.markdown("#### 🤖 ML & Signals")
        st.markdown("ML predictions, PCA factors, data quality")
        st.page_link("pages/2_🤖_ML_Signals.py", label="Open ML →")
    
    with col4:
        st.markdown("#### 💼 Trading")
        st.markdown("Backtest, paper trading, risk dashboard")
        st.page_link("pages/3_💼_Trading.py", label="Open Trading →")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    
    try:
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        
        # Get stock count
        stock_count = db.conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Securities", stock_count)
        with col2:
            st.metric("Data Source", "TradingView")
        with col3:
            st.metric("AI Engine", "Groq (Llama 3.3)")
        with col4:
            st.metric("ML Models", "XGBoost Ensemble")
    except Exception as e:
        st.warning(f"Database not connected: {e}")
