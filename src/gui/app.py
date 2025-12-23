"""
Main application window for MetaQuant Nigeria.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime

try:
    import ttkbootstrap as ttk_bs
    from ttkbootstrap.constants import *
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.database.db_manager import DatabaseManager
from src.gui.theme import apply_theme, COLORS, get_font
from src.gui.tabs.market_intelligence_tab import MarketIntelligenceTab
from src.gui.tabs.screener_tab import ScreenerTab
from src.gui.tabs.portfolio_tab import PortfolioTab
from src.gui.tabs.watchlist_tab import WatchlistTab
from src.gui.tabs.insights_tab import InsightsTab
from src.gui.tabs.universe_tab import UniverseTab
from src.gui.tabs.flow_tab import FlowTab
from src.gui.tabs.history_tab import HistoryTab
from src.gui.tabs.flow_tape_tab import FlowTapeTab
from src.gui.tabs.fundamentals_tab import FundamentalsTab
from src.gui.components.tv_login_dialog import show_tv_login_dialog


logger = logging.getLogger(__name__)


def is_market_open() -> bool:
    """Check if NGX market is currently open.
    
    NGX Trading Hours: Mon-Fri, 10:00 AM - 2:30 PM WAT (GMT+1)
    """
    now = datetime.now()
    
    # Weekend check
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Time check (10:00 AM to 2:30 PM)
    market_open = now.replace(hour=10, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=14, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close


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
        
        # Right side - market status and buttons
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
        
        # Export PDF button
        if TTKBOOTSTRAP_AVAILABLE:
            export_btn = ttk_bs.Button(
                right_frame,
                text="📄 Export PDF",
                bootstyle="info-outline",
                command=self._export_pdf
            )
        else:
            export_btn = ttk.Button(
                right_frame,
                text="📄 Export PDF",
                command=self._export_pdf
            )
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
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
        
        # Settings button (TradingView login)
        if TTKBOOTSTRAP_AVAILABLE:
            settings_btn = ttk_bs.Button(
                right_frame,
                text="⚙️ Settings",
                bootstyle="secondary-outline",
                command=self._show_settings
            )
        else:
            settings_btn = ttk.Button(
                right_frame,
                text="⚙️ Settings",
                command=self._show_settings
            )
        settings_btn.pack(side=tk.LEFT, padx=(10, 0))
    
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
        self.market_intel_tab = MarketIntelligenceTab(self.notebook, self.db)
        self.screener_tab = ScreenerTab(self.notebook, self.db)
        self.portfolio_tab = PortfolioTab(self.notebook, self.db)
        self.watchlist_tab = WatchlistTab(self.notebook, self.db)
        self.insights_tab = InsightsTab(self.notebook, self.db)
        self.universe_tab = UniverseTab(self.notebook, self.db)
        self.flow_tab = FlowTab(self.notebook, self.db)
        self.history_tab = HistoryTab(self.notebook, self.db)
        self.flow_tape_tab = FlowTapeTab(self.notebook, self.db)
        self.fundamentals_tab = FundamentalsTab(self.notebook, self.db)
        
        # Add tabs to notebook
        self.notebook.add(self.market_intel_tab.frame, text="🧠 Market Intel")
        self.notebook.add(self.universe_tab.frame, text="📋 Universe")
        self.notebook.add(self.screener_tab.frame, text="📈 Screener")
        self.notebook.add(self.flow_tab.frame, text="📊 Flow Analysis")
        self.notebook.add(self.flow_tape_tab.frame, text="📊 Flow Tape")
        self.notebook.add(self.fundamentals_tab.frame, text="💰 Fundamentals")
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
    
    def _export_pdf(self):
        """Export daily market summary as PDF."""
        from tkinter import filedialog
        import threading
        
        # Get save location
        default_name = f"MetaQuant_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_name,
            title="Export Market Summary"
        )
        
        if not filepath:
            return
        
        self.set_status("Generating PDF report...")
        
        def generate():
            try:
                from src.reports.report_generator import ReportGenerator, REPORTLAB_AVAILABLE
                from src.collectors.tradingview_collector import TradingViewCollector
                
                if not REPORTLAB_AVAILABLE:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", 
                        "PDF library not installed. Run: pip install reportlab"
                    ))
                    return
                
                # Fetch current market data
                collector = TradingViewCollector()
                all_stocks = collector.get_all_stocks()
                stocks_list = all_stocks.to_dict('records') if not all_stocks.empty else []
                
                # Create snapshot
                gainers = len([s for s in stocks_list if (s.get('change', 0) or 0) > 0])
                losers = len([s for s in stocks_list if (s.get('change', 0) or 0) < 0])
                
                # Get top movers
                sorted_by_change = sorted(stocks_list, key=lambda x: x.get('change', 0) or 0, reverse=True)
                top_gainers = [{'symbol': s.get('symbol'), 'price': s.get('close', 0), 'change': s.get('change', 0)} 
                              for s in sorted_by_change[:5]]
                top_losers = [{'symbol': s.get('symbol'), 'price': s.get('close', 0), 'change': s.get('change', 0)} 
                             for s in sorted_by_change[-5:]]
                
                snapshot = {
                    'total_stocks': len(stocks_list),
                    'gainers': gainers,
                    'losers': losers,
                    'top_gainers': top_gainers,
                    'top_losers': top_losers
                }
                
                # Generate report
                generator = ReportGenerator(self.db)
                success = generator.generate_daily_summary(
                    output_path=filepath,
                    snapshot=snapshot,
                    stocks_list=stocks_list
                )
                
                if success:
                    self.root.after(0, lambda: [
                        self.set_status(f"PDF exported: {filepath}"),
                        messagebox.showinfo("Success", f"Report saved to:\n{filepath}")
                    ])
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate PDF"))
                    
            except Exception as e:
                logger.error(f"PDF export failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Export failed: {str(e)}"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def _on_tab_changed(self, event):
        """Handle tab change events."""
        try:
            current_tab = self.notebook.index(self.notebook.select())
            tab_names = ["Market Intel", "Universe", "Screener", "Flow Analysis", "History", "Portfolio", "Watchlist", "AI Insights"]
            if current_tab < len(tab_names):
                self.set_status(f"Viewing {tab_names[current_tab]}")
        except Exception:
            pass  # Ignore tab change errors
    
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
    
    def _check_market_status(self):
        """Check and update market status."""
        self.update_market_status(is_market_open())
        # Check again in 60 seconds
        self.root.after(60000, self._check_market_status)
    
    def _show_settings(self):
        """Show settings dialog (TradingView login)."""
        def on_login(username, password):
            """Callback when login is attempted."""
            try:
                # Try to create a new TvDatafeed instance with credentials
                from tvDatafeed import TvDatafeed
                tv = TvDatafeed(username=username, password=password)
                
                # Test fetch to verify login
                test_data = tv.get_hist(symbol='DANGCEM', exchange='NSENG', n_bars=5)
                if test_data is not None and not test_data.empty:
                    messagebox.showinfo(
                        "Success",
                        "TradingView login successful!\n\nRestart the app to use premium data."
                    )
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"TradingView login test failed: {e}")
                return False
        
        show_tv_login_dialog(self.root, on_login)
    
    def run(self):
        """Start the application main loop."""
        logger.info("Starting MetaQuant Nigeria")
        # Initial market status check
        self._check_market_status()
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
