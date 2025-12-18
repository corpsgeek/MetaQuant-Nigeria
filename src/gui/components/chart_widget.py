"""Chart widget for portfolio visualization."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.gui.theme import COLORS


class ChartWidget(ttk.Frame):
    """Chart widget using matplotlib."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(self, text="Charts require matplotlib").pack()
            return
        
        # Create figure
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS['bg_dark'])
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _style_axes(self):
        """Apply dark theme to axes."""
        self.ax.set_facecolor(COLORS['bg_medium'])
        self.ax.tick_params(colors=COLORS['text_secondary'])
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.spines['top'].set_color(COLORS['border'])
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['right'].set_color(COLORS['border'])
    
    def plot_pie(self, data: Dict[str, float], title: str = ""):
        """Plot a pie chart."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self._style_axes()
        
        if data:
            labels = list(data.keys())
            sizes = list(data.values())
            self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        
        if title:
            self.ax.set_title(title, color=COLORS['text_primary'])
        
        self.canvas.draw()
    
    def plot_bar(self, labels: List[str], values: List[float], title: str = ""):
        """Plot a bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self._style_axes()
        
        colors = [COLORS['gain'] if v >= 0 else COLORS['loss'] for v in values]
        self.ax.bar(labels, values, color=colors)
        self.ax.axhline(y=0, color=COLORS['border'], linewidth=0.5)
        
        if title:
            self.ax.set_title(title, color=COLORS['text_primary'])
        
        self.canvas.draw()
    
    def clear(self):
        """Clear the chart."""
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self._style_axes()
            self.canvas.draw()
