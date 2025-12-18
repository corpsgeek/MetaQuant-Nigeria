"""
Theme configuration for MetaQuant Nigeria GUI.
Provides a modern dark theme using ttkbootstrap.
"""

import tkinter as tk
from tkinter import ttk

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False


# Color palette
COLORS = {
    # Primary colors
    'primary': '#00D4AA',       # Teal/mint green
    'primary_dark': '#00A88A',
    'primary_light': '#33DFC0',
    
    # Secondary colors
    'secondary': '#6366F1',     # Indigo
    'accent': '#F59E0B',        # Amber/gold
    
    # Background colors
    'bg_dark': '#0F172A',       # Very dark blue
    'bg_medium': '#1E293B',     # Dark slate
    'bg_light': '#334155',      # Lighter slate
    'bg_card': '#1E293B',       # Card background
    
    # Text colors
    'text_primary': '#F8FAFC',  # Almost white
    'text_secondary': '#94A3B8', # Muted gray
    'text_muted': '#64748B',    # More muted
    
    # Status colors
    'success': '#10B981',       # Green
    'danger': '#EF4444',        # Red
    'warning': '#F59E0B',       # Amber
    'info': '#3B82F6',          # Blue
    
    # Gain/Loss colors  
    'gain': '#10B981',          # Green for positive
    'loss': '#EF4444',          # Red for negative
    
    # Border colors
    'border': '#334155',
    'border_light': '#475569',
}

# Typography
FONTS = {
    'heading': ('Inter', 18, 'bold'),
    'subheading': ('Inter', 14, 'bold'),
    'body': ('Inter', 11),
    'body_bold': ('Inter', 11, 'bold'),
    'small': ('Inter', 10),
    'mono': ('JetBrains Mono', 11),
    'mono_small': ('JetBrains Mono', 10),
}

# Fallback fonts if Inter not available
FALLBACK_FONTS = {
    'heading': ('Helvetica Neue', 18, 'bold'),
    'subheading': ('Helvetica Neue', 14, 'bold'),
    'body': ('Helvetica Neue', 11),
    'body_bold': ('Helvetica Neue', 11, 'bold'),
    'small': ('Helvetica Neue', 10),
    'mono': ('Menlo', 11),
    'mono_small': ('Menlo', 10),
}


def get_font(font_key: str) -> tuple:
    """Get font tuple with fallback."""
    try:
        return FONTS.get(font_key, FONTS['body'])
    except:
        return FALLBACK_FONTS.get(font_key, FALLBACK_FONTS['body'])


def apply_theme(root: tk.Tk):
    """Apply custom dark theme to the application."""
    
    if TTKBOOTSTRAP_AVAILABLE:
        # Use ttkbootstrap's dark theme
        return
    
    # Fallback: Configure ttk styles manually
    style = ttk.Style()
    
    # Use clam as base theme (most customizable)
    try:
        style.theme_use('clam')
    except:
        pass
    
    # Configure colors
    style.configure('.', 
        background=COLORS['bg_dark'],
        foreground=COLORS['text_primary'],
        fieldbackground=COLORS['bg_medium'],
        font=get_font('body')
    )
    
    # Frame styles
    style.configure('TFrame', background=COLORS['bg_dark'])
    style.configure('Card.TFrame', background=COLORS['bg_card'])
    
    # Label styles
    style.configure('TLabel', 
        background=COLORS['bg_dark'],
        foreground=COLORS['text_primary']
    )
    style.configure('Heading.TLabel',
        font=get_font('heading'),
        foreground=COLORS['text_primary']
    )
    style.configure('Subheading.TLabel',
        font=get_font('subheading'),
        foreground=COLORS['text_primary']
    )
    style.configure('Muted.TLabel',
        foreground=COLORS['text_muted']
    )
    style.configure('Success.TLabel', foreground=COLORS['success'])
    style.configure('Danger.TLabel', foreground=COLORS['danger'])
    style.configure('Gain.TLabel', foreground=COLORS['gain'])
    style.configure('Loss.TLabel', foreground=COLORS['loss'])
    
    # Button styles
    style.configure('TButton',
        background=COLORS['primary'],
        foreground=COLORS['bg_dark'],
        font=get_font('body_bold'),
        padding=(12, 6)
    )
    style.map('TButton',
        background=[('active', COLORS['primary_dark']),
                   ('pressed', COLORS['primary_dark'])]
    )
    
    style.configure('Secondary.TButton',
        background=COLORS['bg_light'],
        foreground=COLORS['text_primary']
    )
    
    style.configure('Danger.TButton',
        background=COLORS['danger'],
        foreground=COLORS['text_primary']
    )
    
    # Entry styles
    style.configure('TEntry',
        fieldbackground=COLORS['bg_medium'],
        foreground=COLORS['text_primary'],
        insertcolor=COLORS['text_primary'],
        padding=8
    )
    
    # Combobox styles
    style.configure('TCombobox',
        fieldbackground=COLORS['bg_medium'],
        background=COLORS['bg_light'],
        foreground=COLORS['text_primary'],
        arrowcolor=COLORS['text_primary']
    )
    
    # Notebook (tabs) styles
    style.configure('TNotebook',
        background=COLORS['bg_dark'],
        tabmargins=[0, 0, 0, 0]
    )
    style.configure('TNotebook.Tab',
        background=COLORS['bg_medium'],
        foreground=COLORS['text_secondary'],
        padding=[16, 8],
        font=get_font('body_bold')
    )
    style.map('TNotebook.Tab',
        background=[('selected', COLORS['bg_dark'])],
        foreground=[('selected', COLORS['primary'])]
    )
    
    # Treeview styles
    style.configure('Treeview',
        background=COLORS['bg_medium'],
        foreground=COLORS['text_primary'],
        fieldbackground=COLORS['bg_medium'],
        rowheight=32,
        font=get_font('body')
    )
    style.configure('Treeview.Heading',
        background=COLORS['bg_light'],
        foreground=COLORS['text_primary'],
        font=get_font('body_bold')
    )
    style.map('Treeview',
        background=[('selected', COLORS['primary_dark'])],
        foreground=[('selected', COLORS['text_primary'])]
    )
    
    # Scrollbar styles
    style.configure('TScrollbar',
        background=COLORS['bg_light'],
        troughcolor=COLORS['bg_dark'],
        arrowcolor=COLORS['text_secondary']
    )
    
    # Progressbar styles
    style.configure('TProgressbar',
        background=COLORS['primary'],
        troughcolor=COLORS['bg_medium']
    )
    
    # LabelFrame styles
    style.configure('TLabelframe',
        background=COLORS['bg_dark']
    )
    style.configure('TLabelframe.Label',
        background=COLORS['bg_dark'],
        foreground=COLORS['text_primary'],
        font=get_font('subheading')
    )


def format_currency(value: float, symbol: str = "₦") -> str:
    """Format a number as Nigerian Naira."""
    if value is None:
        return "N/A"
    return f"{symbol}{value:,.2f}"


def format_percent(value: float, include_sign: bool = True) -> str:
    """Format a number as percentage."""
    if value is None:
        return "N/A"
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"₦{value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"₦{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"₦{value/1_000:.2f}K"
    return f"₦{value:.2f}"


def get_change_color(value: float) -> str:
    """Get color based on positive/negative value."""
    if value is None:
        return COLORS['text_muted']
    if value > 0:
        return COLORS['gain']
    if value < 0:
        return COLORS['loss']
    return COLORS['text_secondary']
