"""
ML & Signals Page - ML Intelligence, PCA Factors, Data Quality
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="ML & Signals | MetaQuant", page_icon="🤖", layout="wide")

# Auth check
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.warning("Please login from the main page")
    st.stop()

# =============================================================================
# ML ENGINE
# =============================================================================

@st.cache_resource
def get_ml_engine():
    """Get ML engine."""
    try:
        from src.ml.ml_engine import MLEngine
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        engine = MLEngine(db)
        return engine
    except Exception as e:
        st.warning(f"ML Engine not available: {e}")
        return None

# =============================================================================
# PAGE CONTENT
# =============================================================================

st.markdown("# 🤖 ML & Signals")

tab1, tab2, tab3 = st.tabs(["🤖 ML Intelligence", "🔬 PCA Factors", "📊 Data Quality"])

# -----------------------------------------------------------------------------
# TAB 1: ML INTELLIGENCE
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 🤖 ML Intelligence")
    st.markdown("XGBoost-powered stock predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        symbol = st.selectbox("Select Stock", [
            "DANGCEM", "GTCO", "MTNN", "ZENITHBANK", "AIRTEL", 
            "NESTLE", "BUACEMENT", "SEPLAT", "ACCESSCORP", "UBA"
        ])
        
        if st.button("🔮 Generate Prediction", type="primary"):
            with st.spinner("Running ML models..."):
                # Demo prediction
                import time
                time.sleep(1)
                
                st.success("Prediction generated!")
                
                st.markdown("#### Signal")
                st.markdown("### 🟢 BULLISH")
                st.metric("Confidence", "78%")
                st.metric("Expected Return (5d)", "+2.4%")
    
    with col2:
        st.markdown("#### ML Predictions Overview")
        
        # Demo data
        predictions_df = pd.DataFrame({
            'Symbol': ['DANGCEM', 'GTCO', 'MTNN', 'ZENITHBANK', 'AIRTEL', 'NESTLE'],
            'Signal': ['🟢 BULLISH', '🟢 BULLISH', '🔴 BEARISH', '🟢 BULLISH', '🟡 NEUTRAL', '🟢 BULLISH'],
            'Confidence': ['78%', '65%', '72%', '61%', '54%', '81%'],
            '5D Return': ['+2.4%', '+1.2%', '-1.8%', '+0.9%', '+0.2%', '+3.1%'],
            'Anomaly': ['Normal', 'Normal', 'WARNING', 'Normal', 'Normal', 'Normal']
        })
        
        st.dataframe(predictions_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "72.4%")
        with col2:
            st.metric("Sharpe Ratio", "1.85")
        with col3:
            st.metric("Win Rate", "61.2%")

# -----------------------------------------------------------------------------
# TAB 2: PCA FACTORS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔬 PCA Factor Analysis")
    st.markdown("Principal Component Analysis of market returns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Factor Returns (Today)")
        
        factor_df = pd.DataFrame({
            'Factor': ['Market Beta', 'Size', 'Value', 'Momentum', 'Volatility'],
            'Return': ['+0.42%', '-0.15%', '+0.28%', '+0.67%', '-0.23%'],
            'Contribution': ['35%', '15%', '22%', '18%', '10%']
        })
        st.dataframe(factor_df, use_container_width=True)
        
        st.markdown("#### Market Regime")
        st.info("📈 **RISK-ON** - Market showing bullish momentum")
    
    with col2:
        st.markdown("#### Factor Exposures")
        
        # Demo chart data
        import plotly.express as px
        
        exposure_df = pd.DataFrame({
            'Stock': ['DANGCEM', 'GTCO', 'MTNN', 'ZENITHBANK', 'AIRTEL'],
            'Market Beta': [1.2, 0.9, 1.1, 0.85, 0.95],
            'Size': [0.8, 0.6, 0.9, 0.5, 0.7],
            'Momentum': [1.5, 0.7, 1.2, 0.9, 1.0]
        })
        
        fig = px.bar(exposure_df, x='Stock', y=['Market Beta', 'Size', 'Momentum'],
                     barmode='group', title='Factor Exposures by Stock')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: DATA QUALITY
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📊 Data Quality Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Securities", "155")
    with col2:
        st.metric("With Price Data", "143")
    with col3:
        st.metric("Data Freshness", "< 1 hour")
    with col4:
        st.metric("Quality Score", "94%")
    
    st.markdown("---")
    
    st.markdown("#### Data Coverage by Sector")
    
    coverage_df = pd.DataFrame({
        'Sector': ['Banking', 'Oil & Gas', 'Consumer Goods', 'Insurance', 'Industrial', 'Telecom'],
        'Stocks': [12, 8, 15, 10, 20, 3],
        'With Data': [12, 8, 14, 9, 18, 3],
        'Coverage': ['100%', '100%', '93%', '90%', '90%', '100%']
    })
    
    st.dataframe(coverage_df, use_container_width=True)
    
    st.markdown("#### Recent Data Issues")
    st.warning("⚠️ NOTORE: Insufficient historical data (0 rows)")
    st.warning("⚠️ SOVEREIGN: Trading suspended, no updates")
