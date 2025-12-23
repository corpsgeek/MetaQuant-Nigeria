"""
TradingView Login Dialog for MetaQuant Nigeria.
Allows user to input TradingView credentials for premium data access.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import os

try:
    import ttkbootstrap as ttk_bs
    TTKBOOTSTRAP_AVAILABLE = True
except ImportError:
    TTKBOOTSTRAP_AVAILABLE = False

from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class TradingViewLoginDialog:
    """Dialog for TradingView login credentials."""
    
    def __init__(self, parent, on_login_callback=None):
        """
        Initialize the login dialog.
        
        Args:
            parent: Parent window
            on_login_callback: Function to call with (username, password) on successful login attempt
        """
        self.parent = parent
        self.on_login_callback = on_login_callback
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("TradingView Login")
        self.dialog.geometry("400x280")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 280) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self._create_ui()
        
        # Load existing credentials if any
        self._load_existing_credentials()
    
    def _create_ui(self):
        """Create the login UI."""
        # Main container
        main = ttk.Frame(self.dialog, padding=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(
            main,
            text="🔐 TradingView Login",
            font=get_font('subheading'),
            foreground=COLORS['primary']
        ).pack(pady=(0, 10))
        
        ttk.Label(
            main,
            text="Enter your TradingView credentials for premium data access.",
            font=get_font('small'),
            foreground=COLORS['text_muted'],
            wraplength=350
        ).pack(pady=(0, 15))
        
        # Username
        username_frame = ttk.Frame(main)
        username_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            username_frame,
            text="Username/Email:",
            font=get_font('body')
        ).pack(anchor='w')
        
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(
            username_frame,
            textvariable=self.username_var,
            width=40
        )
        self.username_entry.pack(fill=tk.X, pady=(3, 0))
        
        # Password
        password_frame = ttk.Frame(main)
        password_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            password_frame,
            text="Password:",
            font=get_font('body')
        ).pack(anchor='w')
        
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(
            password_frame,
            textvariable=self.password_var,
            show="•",
            width=40
        )
        self.password_entry.pack(fill=tk.X, pady=(3, 0))
        
        # Save to .env checkbox
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            main,
            text="Save credentials to .env file",
            variable=self.save_var
        ).pack(anchor='w', pady=(5, 10))
        
        # Status label
        self.status_label = ttk.Label(
            main,
            text="",
            font=get_font('small'),
            foreground=COLORS['text_muted']
        )
        self.status_label.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text="🔑 Login",
            command=self._on_login,
            style="Accent.TButton" if TTKBOOTSTRAP_AVAILABLE else None
        ).pack(side=tk.RIGHT)
        
        # Bind Enter key
        self.dialog.bind('<Return>', lambda e: self._on_login())
        self.username_entry.focus_set()
    
    def _load_existing_credentials(self):
        """Load existing credentials from environment if available."""
        username = os.getenv('TV_USERNAME', '')
        password = os.getenv('TV_PASSWORD', '')
        
        if username:
            self.username_var.set(username)
        if password:
            self.password_var.set(password)
            self.status_label.config(
                text="✓ Existing credentials found",
                foreground=COLORS['gain']
            )
    
    def _on_login(self):
        """Handle login button click."""
        username = self.username_var.get().strip()
        password = self.password_var.get()
        
        if not username or not password:
            self.status_label.config(
                text="⚠️ Please enter both username and password",
                foreground=COLORS['warning']
            )
            return
        
        self.status_label.config(
            text="🔄 Attempting login...",
            foreground=COLORS['text_muted']
        )
        self.dialog.update()
        
        # Save to .env if requested
        if self.save_var.get():
            self._save_to_env(username, password)
        
        # Set environment variables for current session
        os.environ['TV_USERNAME'] = username
        os.environ['TV_PASSWORD'] = password
        
        self.result = (username, password)
        
        # Call callback if provided
        if self.on_login_callback:
            try:
                success = self.on_login_callback(username, password)
                if success:
                    self.status_label.config(
                        text="✓ Login successful!",
                        foreground=COLORS['gain']
                    )
                    self.dialog.after(1000, self.dialog.destroy)
                else:
                    self.status_label.config(
                        text="⚠️ Login failed - credentials saved for retry",
                        foreground=COLORS['warning']
                    )
            except Exception as e:
                self.status_label.config(
                    text=f"⚠️ Error: {str(e)[:40]}",
                    foreground=COLORS['loss']
                )
        else:
            self.dialog.destroy()
    
    def _save_to_env(self, username: str, password: str):
        """Save credentials to .env file."""
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env')
            
            # Read existing .env content
            existing_lines = []
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    existing_lines = f.readlines()
            
            # Update or add TV credentials
            tv_username_found = False
            tv_password_found = False
            new_lines = []
            
            for line in existing_lines:
                if line.startswith('TV_USERNAME='):
                    new_lines.append(f'TV_USERNAME={username}\n')
                    tv_username_found = True
                elif line.startswith('TV_PASSWORD='):
                    new_lines.append(f'TV_PASSWORD={password}\n')
                    tv_password_found = True
                else:
                    new_lines.append(line)
            
            # Add if not found
            if not tv_username_found:
                new_lines.append(f'TV_USERNAME={username}\n')
            if not tv_password_found:
                new_lines.append(f'TV_PASSWORD={password}\n')
            
            # Write back
            with open(env_path, 'w') as f:
                f.writelines(new_lines)
            
            logger.info("Saved TradingView credentials to .env")
            
        except Exception as e:
            logger.error(f"Failed to save credentials to .env: {e}")
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog and wait for it to close."""
        self.dialog.wait_window()
        return self.result


def show_tv_login_dialog(parent, on_login_callback=None):
    """
    Show the TradingView login dialog.
    
    Args:
        parent: Parent window
        on_login_callback: Optional callback(username, password) -> bool
        
    Returns:
        (username, password) tuple or None if cancelled
    """
    dialog = TradingViewLoginDialog(parent, on_login_callback)
    return dialog.show()
