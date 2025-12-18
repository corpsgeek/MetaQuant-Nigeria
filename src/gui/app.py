"""
Main application window for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import apply_theme, COLORS, get_font
from src.gui.tabs.screener_tab import ScreenerTab
from src.gui.tabs.portfolio_tab import PortfolioTab
from src.gui.tabs.watchlist_tab import WatchlistTab
from src.gui.tabs.insights_tab import InsightsTab
from src.gui.tabs.universe_tab import UniverseTab
from src.gui.tabs.flow_tab import FlowTab
from src.gui.tabs.history_tab import HistoryTab


logger = logging.getLogger(__name__)


class MetaQuantApp:
    """
    Main application class for MetaQuant Nigeria Stock Screener.
    """
    
    APP_NAME = "MetaQuant Nigeria"
    APP_VERSION = "0.1.0"
    DEFAULT_WIDTH = 1400
    DEFAULT_HEIGHT = 900
    MIN_WIDTH = 1000
    MIN_HEIGHT = 700
    
    def __init__(self, db: DatabaseManager):
        """
        Initialize the application.
        
        Args:
            db: Database manager instance
        """
        self.db = db
        
        # Create main window
        if TTKBOOTSTRAP_AVAILABLE:
            self.root = ttk_bs.Window(
                title=f"{self.APP_NAME} v{self.APP_VERSION}",
                themename="darkly",  # Dark theme
                size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT),
                minsize=(self.MIN_WIDTH, self.MIN_HEIGHT)
            )
        else:
            self.root = tk.Tk()
            self.root.title(f"{self.APP_NAME} v{self.APP_VERSION}")
            self.root.geometry(f"{self.DEFAULT_WIDTH}x{self.DEFAULT_HEIGHT}")
            self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
            apply_theme(self.root)
        
        # Configure root
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Setup UI
        self._setup_ui()
        
        # Bind events
        self._bind_events()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._create_header()
        
        # Notebook (tabbed interface)
        self._create_notebook()
        
        # Status bar
        self._create_status_bar()
    
    def _create_header(self):
        """Create the application header."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, padx=20, pady=(15, 5))
        
        # Logo and title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT)
        
        title_label = ttk.Label(
            title_frame,
            text="📊 MetaQuant",
            font=get_font('heading'),
            foreground=COLORS['primary']
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Nigeria Stock Screener",
            font=get_font('body'),
            foreground=COLORS['text_secondary']
        )
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Right side - market status and refresh
        right_frame = ttk.Frame(header_frame)
        right_frame.pack(side=tk.RIGHT)
        
        # Market status indicator
        self.market_status_label = ttk.Label(
            right_frame,
            text="● Market Closed",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.market_status_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Refresh button
        if TTKBOOTSTRAP_AVAILABLE:
            refresh_btn = ttk_bs.Button(
                right_frame,
                text="↻ Refresh Data",
                bootstyle="outline",
                command=self._refresh_data
            )
        else:
            refresh_btn = ttk.Button(
                right_frame,
                text="↻ Refresh Data",
                command=self._refresh_data
            )
        refresh_btn.pack(side=tk.LEFT)
    
    def _create_notebook(self):
        """Create the tabbed notebook interface."""
        notebook_frame = ttk.Frame(self.main_frame)
        notebook_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        if TTKBOOTSTRAP_AVAILABLE:
            self.notebook = ttk_bs.Notebook(notebook_frame, bootstyle="dark")
        else:
            self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.screener_tab = ScreenerTab(self.notebook, self.db)
        self.portfolio_tab = PortfolioTab(self.notebook, self.db)
        self.watchlist_tab = WatchlistTab(self.notebook, self.db)
        self.insights_tab = InsightsTab(self.notebook, self.db)
        self.universe_tab = UniverseTab(self.notebook, self.db)
        self.flow_tab = FlowTab(self.notebook, self.db)
        self.history_tab = HistoryTab(self.notebook, self.db)
        
        # Add tabs to notebook
        self.notebook.add(self.universe_tab.frame, text="📋 Universe")
        self.notebook.add(self.screener_tab.frame, text="📈 Screener")
        self.notebook.add(self.flow_tab.frame, text="📊 Flow Analysis")
        self.notebook.add(self.history_tab.frame, text="📅 History")
        self.notebook.add(self.portfolio_tab.frame, text="💼 Portfolio")
        self.notebook.add(self.watchlist_tab.frame, text="👁 Watchlist")
        self.notebook.add(self.insights_tab.frame, text="🤖 AI Insights")
    
    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Separator line
        separator = ttk.Separator(status_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X)
        
        # Status content
        status_content = ttk.Frame(status_frame)
        status_content.pack(fill=tk.X, padx=20, pady=8)
        
        # Left: Status message
        self.status_label = ttk.Label(
            status_content,
            text="Ready",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Right: Database info
        db_label = ttk.Label(
            status_content,
            text=f"Database: {self.db.db_path}",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        db_label.pack(side=tk.RIGHT)
    
    def _bind_events(self):
        """Bind keyboard shortcuts and events."""
        # Keyboard shortcuts
        self.root.bind('<Control-r>', lambda e: self._refresh_data())
        self.root.bind('<Control-q>', lambda e: self._quit())
        self.root.bind('<F5>', lambda e: self._refresh_data())
        
        # Tab change
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
        # Window close
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
    
    def _refresh_data(self):
        """Refresh data from collectors."""
        self.set_status("Refreshing data...")
        
        try:
            # Refresh the active tab
            current_tab = self.notebook.index(self.notebook.select())
            
            if current_tab == 0:
                self.screener_tab.refresh()
            elif current_tab == 1:
                self.portfolio_tab.refresh()
            elif current_tab == 2:
                self.watchlist_tab.refresh()
            elif current_tab == 3:
                self.insights_tab.refresh()
            
            self.set_status("Data refreshed")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")
            self.set_status(f"Refresh failed: {str(e)}")
    
    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.notebook.index(self.notebook.select())
        tab_names = ["Universe", "Screener", "Portfolio", "Watchlist", "AI Insights"]
        self.set_status(f"Viewing {tab_names[current_tab]}")
    
    def _quit(self):
        """Clean shutdown of the application."""
        try:
            self.db.close()
        except:
            pass
        self.root.destroy()
    
    def set_status(self, message: str):
        """Update the status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_market_status(self, is_open: bool):
        """Update the market status indicator."""
        if is_open:
            self.market_status_label.config(
                text="● Market Open",
                foreground=COLORS['success']
            )
        else:
            self.market_status_label.config(
                text="● Market Closed",
                foreground=COLORS['text_muted']
            )
    
    def run(self):
        """Start the application main loop."""
        logger.info("Starting MetaQuant Nigeria")
        self.root.mainloop()


def main():
    """Entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    db = DatabaseManager()
    db.initialize()
    
    app = MetaQuantApp(db)
    app.run()


if __name__ == "__main__":
    main()
