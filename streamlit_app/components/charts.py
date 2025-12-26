"""
Shared Chart Components for Streamlit App
Plotly-based chart helpers replicating tkinter Canvas visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def create_price_chart(df: pd.DataFrame, symbol: str = '') -> go.Figure:
    """Create OHLC candlestick chart with volume."""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index if 'datetime' not in df.columns else df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Price Chart',
        xaxis_rangeslider_visible=False,
        height=500,
        template='plotly_dark'
    )
    
    return fig


def create_vwap_chart(df: pd.DataFrame, symbol: str = '') -> go.Figure:
    """Create VWAP chart with deviation bands."""
    if df.empty or 'close' not in df.columns:
        return go.Figure()
    
    # Calculate VWAP
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_vol_price'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    
    # Calculate deviation
    df['deviation'] = df['close'] - df['vwap']
    std = df['deviation'].std()
    df['upper_1'] = df['vwap'] + std
    df['upper_2'] = df['vwap'] + 2 * std
    df['lower_1'] = df['vwap'] - std
    df['lower_2'] = df['vwap'] - 2 * std
    
    fig = go.Figure()
    
    # Fill bands
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_2'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_2'], fill='tonexty', fillcolor='rgba(100,100,100,0.2)', 
                              line=dict(width=0), name='±2 Std'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_1'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_1'], fill='tonexty', fillcolor='rgba(100,100,100,0.3)', 
                              line=dict(width=0), name='±1 Std'))
    
    # VWAP line
    fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='blue', width=2), name='VWAP'))
    
    # Price line
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], line=dict(color='white', width=1), name='Price'))
    
    fig.update_layout(
        title=f'{symbol} VWAP Analysis',
        height=400,
        template='plotly_dark'
    )
    
    return fig


def create_momentum_chart(df: pd.DataFrame) -> go.Figure:
    """Create momentum/delta oscillator chart."""
    if df.empty or 'close' not in df.columns:
        return go.Figure()
    
    df = df.copy()
    df['momentum'] = df['close'].pct_change(5) * 100  # 5-period momentum
    df['momentum_sma'] = df['momentum'].rolling(10).mean()
    
    fig = go.Figure()
    
    # Momentum bars
    colors = ['green' if m > 0 else 'red' for m in df['momentum']]
    fig.add_trace(go.Bar(x=df.index, y=df['momentum'], marker_color=colors, name='Momentum'))
    
    # Signal line
    fig.add_trace(go.Scatter(x=df.index, y=df['momentum_sma'], line=dict(color='yellow', width=2), 
                              name='Signal'))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Momentum Oscillator',
        height=300,
        template='plotly_dark'
    )
    
    return fig


def create_volume_profile(df: pd.DataFrame, bins: int = 20) -> go.Figure:
    """Create volume profile chart (horizontal histogram)."""
    if df.empty or 'close' not in df.columns:
        return go.Figure()
    
    # Create price bins
    price_min, price_max = df['close'].min(), df['close'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)
    df['price_bin'] = pd.cut(df['close'], bins=price_bins)
    
    # Aggregate volume by price bin
    vol_profile = df.groupby('price_bin')['volume'].sum().reset_index()
    vol_profile['price_mid'] = vol_profile['price_bin'].apply(lambda x: x.mid if pd.notna(x) else 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=vol_profile['price_mid'],
        x=vol_profile['volume'],
        orientation='h',
        marker_color='rgba(0, 150, 255, 0.7)',
        name='Volume'
    ))
    
    fig.update_layout(
        title='Volume Profile',
        xaxis_title='Volume',
        yaxis_title='Price',
        height=400,
        template='plotly_dark'
    )
    
    return fig


def create_sector_heatmap(sector_data: List[Dict]) -> go.Figure:
    """Create sector performance heatmap."""
    if not sector_data:
        return go.Figure()
    
    df = pd.DataFrame(sector_data)
    
    fig = px.treemap(
        df,
        path=['sector'],
        values='count',
        color='avg_1d',
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Sector Performance Heatmap'
    )
    
    fig.update_layout(height=400, template='plotly_dark')
    return fig


def create_sector_bar_chart(sector_data: List[Dict]) -> go.Figure:
    """Create sector performance bar chart."""
    if not sector_data:
        return go.Figure()
    
    df = pd.DataFrame(sector_data)
    
    colors = ['green' if x > 0 else 'red' for x in df['avg_1d']]
    
    fig = go.Figure(go.Bar(
        x=df['sector'],
        y=df['avg_1d'],
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in df['avg_1d']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Sector 1-Day Performance',
        xaxis_title='Sector',
        yaxis_title='Change %',
        height=400,
        template='plotly_dark'
    )
    
    return fig


def create_flow_gauge(buying_pct: float, selling_pct: float) -> go.Figure:
    """Create money flow pressure gauge."""
    net = buying_pct - selling_pct
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=net,
        delta={'reference': 0},
        title={'text': "Net Flow Pressure"},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [-100, -20], 'color': "red"},
                {'range': [-20, 20], 'color': "gray"},
                {'range': [20, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "yellow", 'width': 4},
                'thickness': 0.75,
                'value': net
            }
        }
    ))
    
    fig.update_layout(height=300, template='plotly_dark')
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None) -> go.Figure:
    """Create correlation matrix heatmap."""
    if df.empty:
        return go.Figure()
    
    if columns:
        df = df[columns]
    
    corr = df.corr()
    
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=500,
        template='plotly_dark'
    )
    
    return fig


def create_factor_chart(factors: List[Dict]) -> go.Figure:
    """Create factor returns bar chart."""
    if not factors:
        return go.Figure()
    
    df = pd.DataFrame(factors)
    
    colors = ['green' if r > 0 else 'red' for r in df.get('return', [0] * len(df))]
    
    fig = go.Figure(go.Bar(
        x=df.get('name', df.index),
        y=df.get('return', [0] * len(df)),
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Factor Returns',
        height=300,
        template='plotly_dark'
    )
    
    return fig


def create_prediction_chart(predictions: List[Dict]) -> go.Figure:
    """Create ML predictions summary chart."""
    if not predictions:
        return go.Figure()
    
    df = pd.DataFrame(predictions)
    
    signal_counts = df['signal'].value_counts()
    
    colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
    
    fig = go.Figure(go.Bar(
        x=signal_counts.index,
        y=signal_counts.values,
        marker_color=[colors.get(s, 'blue') for s in signal_counts.index]
    ))
    
    fig.update_layout(
        title='ML Signal Distribution',
        height=300,
        template='plotly_dark'
    )
    
    return fig


def create_rsi_chart(df: pd.DataFrame, period: int = 14) -> go.Figure:
    """Create RSI indicator chart."""
    if df.empty or 'close' not in df.columns:
        return go.Figure()
    
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='purple', width=2), name='RSI'))
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title='RSI Indicator',
        yaxis=dict(range=[0, 100]),
        height=250,
        template='plotly_dark'
    )
    
    return fig
