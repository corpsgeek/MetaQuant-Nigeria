"""Reusable filter widget."""

import tkinter as tk
from tkinter import ttk


class FilterWidget(ttk.LabelFrame):
    """Reusable filter input widget."""
    
    def __init__(self, parent, label: str, filter_type: str = 'range', **kwargs):
        super().__init__(parent, text=label, padding=5, **kwargs)
        
        self.filter_type = filter_type
        self.min_var = tk.StringVar()
        self.max_var = tk.StringVar()
        self.value_var = tk.StringVar()
        
        self._setup_inputs()
    
    def _setup_inputs(self):
        """Setup input fields based on type."""
        if self.filter_type == 'range':
            ttk.Entry(self, textvariable=self.min_var, width=8).pack(side=tk.LEFT, padx=2)
            ttk.Label(self, text="-").pack(side=tk.LEFT)
            ttk.Entry(self, textvariable=self.max_var, width=8).pack(side=tk.LEFT, padx=2)
        
        elif self.filter_type == 'min':
            ttk.Label(self, text="Min:").pack(side=tk.LEFT)
            ttk.Entry(self, textvariable=self.min_var, width=8).pack(side=tk.LEFT, padx=5)
        
        elif self.filter_type == 'max':
            ttk.Label(self, text="Max:").pack(side=tk.LEFT)
            ttk.Entry(self, textvariable=self.max_var, width=8).pack(side=tk.LEFT, padx=5)
        
        elif self.filter_type == 'text':
            ttk.Entry(self, textvariable=self.value_var, width=15).pack(fill=tk.X)
    
    def get_min(self):
        """Get minimum value."""
        try:
            return float(self.min_var.get()) if self.min_var.get() else None
        except ValueError:
            return None
    
    def get_max(self):
        """Get maximum value."""
        try:
            return float(self.max_var.get()) if self.max_var.get() else None
        except ValueError:
            return None
    
    def get_value(self):
        """Get text value."""
        return self.value_var.get().strip() or None
    
    def clear(self):
        """Clear all inputs."""
        self.min_var.set('')
        self.max_var.set('')
        self.value_var.set('')
