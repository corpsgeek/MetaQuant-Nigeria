"""
Shared Metric Components for Streamlit App
Reusable metric cards and display helpers
"""

import streamlit as st
from typing import Dict, List, Optional, Union


def metric_card(label: str, value: Union[str, float, int], delta: Optional[str] = None, 
                color: str = None, help_text: str = None):
    """Display a styled metric card."""
    if isinstance(value, float):
        if abs(value) < 1:
            value = f"{value:.2%}"
        else:
            value = f"{value:,.2f}"
    elif isinstance(value, int):
        value = f"{value:,}"
    
    if delta and color:
        # Custom colored metric
        st.metric(label, value, delta=delta, help=help_text)
    elif delta:
        st.metric(label, value, delta=delta, help=help_text)
    else:
        st.metric(label, value, help=help_text)


def signal_badge(signal: str, confidence: float = None):
    """Display a signal badge (BUY/SELL/HOLD)."""
    badges = {
        'BUY': ('🟢', 'BULLISH', 'green'),
        'STRONG_BUY': ('🟢', 'STRONG BUY', 'green'),
        'SELL': ('🔴', 'BEARISH', 'red'),
        'STRONG_SELL': ('🔴', 'STRONG SELL', 'red'),
        'HOLD': ('🟡', 'NEUTRAL', 'orange'),
    }
    
    emoji, text, color = badges.get(signal.upper(), ('⚪', signal, 'gray'))
    
    conf_text = f" ({confidence:.0%})" if confidence else ""
    st.markdown(f"### {emoji} {text}{conf_text}")


def regime_indicator(regime: str, confidence: float = None):
    """Display market regime indicator."""
    regimes = {
        'RISK_ON': ('🟢', 'RISK-ON', 'Bullish momentum'),
        'RISK_OFF': ('🔴', 'RISK-OFF', 'Defensive positioning'),
        'NEUTRAL': ('🟡', 'NEUTRAL', 'Mixed signals'),
        'UNKNOWN': ('⚪', 'UNKNOWN', 'Insufficient data'),
    }
    
    emoji, label, desc = regimes.get(regime.upper(), ('⚪', regime, ''))
    
    conf_text = f" • {confidence:.0%} confidence" if confidence else ""
    
    st.info(f"{emoji} **{label}**{conf_text}\n\n{desc}")


def flow_cards(strong_in: int, mod_in: int, neutral: int, mod_out: int, strong_out: int):
    """Display flow classification cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🚀 Strong In", strong_in, help="+5%+ (1W)")
    with col2:
        st.metric("📈 Inflow", mod_in, help="+0-5% (1W)")
    with col3:
        st.metric("➡️ Neutral", neutral, help="±1% (1W)")
    with col4:
        st.metric("📉 Outflow", mod_out, help="-0-5% (1W)")
    with col5:
        st.metric("💨 Strong Out", strong_out, help="-5%+ (1W)")


def breadth_cards(gainers: int, losers: int, unchanged: int, avg_change: float = None):
    """Display market breadth cards."""
    total = gainers + losers + unchanged
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pct = f"{gainers/total*100:.1f}%" if total else "0%"
        st.metric("📈 Gainers", gainers, delta=pct)
    with col2:
        pct = f"-{losers/total*100:.1f}%" if total else "0%"
        st.metric("📉 Losers", losers, delta=pct)
    with col3:
        st.metric("➡️ Unchanged", unchanged)
    with col4:
        if avg_change is not None:
            st.metric("📊 Avg Change", f"{avg_change:+.2f}%")


def valuation_card(label: str, current: float, avg: float, percentile: float = None):
    """Display valuation metric card with comparison."""
    vs_avg = (current / avg - 1) * 100 if avg else 0
    
    if vs_avg > 20:
        status = "🔴 Overvalued"
    elif vs_avg < -20:
        status = "🟢 Undervalued"
    else:
        status = "🟡 Fair Value"
    
    st.markdown(f"**{label}**")
    st.markdown(f"### {current:.2f}x")
    st.markdown(f"Sector Avg: {avg:.2f}x ({vs_avg:+.1f}%)")
    st.markdown(f"**{status}**")
    
    if percentile is not None:
        st.progress(percentile / 100)
        st.caption(f"{percentile:.0f}th percentile")


def performance_table(stocks: List[Dict], columns: List[str] = None):
    """Display a performance table for stocks."""
    import pandas as pd
    
    if not stocks:
        st.info("No data available")
        return
    
    df = pd.DataFrame(stocks)
    
    if columns:
        df = df[columns]
    
    # Format percentage columns
    for col in df.columns:
        if 'chg' in col.lower() or 'return' in col.lower() or 'perf' in col.lower():
            df[col] = df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "--")
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def sector_rankings_table(sector_data: List[Dict]):
    """Display sector rankings table."""
    import pandas as pd
    
    if not sector_data:
        st.info("No sector data")
        return
    
    df = pd.DataFrame(sector_data)
    
    display = df[['sector', 'avg_1d', 'avg_1w', 'count', 'gainers', 'losers']].copy()
    display['avg_1d'] = display['avg_1d'].apply(lambda x: f"{x:+.2f}%")
    display['avg_1w'] = display['avg_1w'].apply(lambda x: f"{x:+.2f}%")
    display.columns = ['Sector', '1D Chg', '1W Chg', 'Stocks', '↑', '↓']
    
    st.dataframe(display, use_container_width=True, hide_index=True)


def prediction_table(predictions: List[Dict]):
    """Display ML predictions table."""
    import pandas as pd
    
    if not predictions:
        st.info("No predictions available")
        return
    
    df = pd.DataFrame(predictions)
    
    # Add signal emoji
    signal_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
    df['signal_icon'] = df['signal'].map(signal_map).fillna('⚪') + ' ' + df['signal']
    
    # Format columns
    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.0%}")
    df['expected_return'] = df['expected_return'].apply(lambda x: f"{x:+.2f}%")
    
    display = df[['symbol', 'signal_icon', 'confidence', 'expected_return']]
    display.columns = ['Symbol', 'Signal', 'Confidence', 'Exp. Return']
    
    st.dataframe(display, use_container_width=True, hide_index=True)


def alert_box(title: str, message: str, alert_type: str = 'info'):
    """Display an alert box."""
    if alert_type == 'success':
        st.success(f"**{title}**\n\n{message}")
    elif alert_type == 'warning':
        st.warning(f"**{title}**\n\n{message}")
    elif alert_type == 'error':
        st.error(f"**{title}**\n\n{message}")
    else:
        st.info(f"**{title}**\n\n{message}")


def loading_placeholder(text: str = "Loading data..."):
    """Display a loading placeholder."""
    st.info(f"⏳ {text}")
