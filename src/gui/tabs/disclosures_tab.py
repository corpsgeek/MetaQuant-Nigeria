"""
Corporate Disclosures Tab for MetaQuant Nigeria.
Displays NGX corporate disclosures with AI-powered analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, List, Optional
import threading
import webbrowser
from datetime import datetime

from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class DisclosuresTab:
    """
    Corporate Disclosures Intelligence Tab.
    
    Features:
    - List of recent disclosures with company/title/date
    - AI-generated summaries and impact scores
    - Filter by company, impact level
    - Click to view details and open PDF
    """
    
    def __init__(self, parent, db):
        self.parent = parent
        self.db = db
        self.frame = ttk.Frame(parent)
        
        # Initialize components
        self._init_components()
        self._setup_ui()
        
        # Load disclosures on startup (delayed)
        self.frame.after(1000, self._refresh_disclosures)
    
    def _init_components(self):
        """Initialize scraper, parser, and analyzer."""
        try:
            from src.collectors.disclosure_scraper import DisclosureScraper
            self.scraper = DisclosureScraper(self.db)
        except Exception as e:
            logger.error(f"Failed to init disclosure scraper: {e}")
            self.scraper = None
        
        try:
            from src.collectors.pdf_parser import PDFParser
            self.pdf_parser = PDFParser()
        except Exception as e:
            logger.error(f"Failed to init PDF parser: {e}")
            self.pdf_parser = None
        
        try:
            from src.ai.disclosure_analyzer import DisclosureAnalyzer
            self.analyzer = DisclosureAnalyzer()
        except Exception as e:
            logger.error(f"Failed to init disclosure analyzer: {e}")
            self.analyzer = None
        
        self.disclosures = []
        self.selected_disclosure = None
    
    def _setup_ui(self):
        """Setup the UI components."""
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # === HEADER ===
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header, text="📋 Corporate Disclosures Intelligence",
                 font=get_font('heading'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="🔄 Sync NGX", 
                  command=self._sync_disclosures).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🤖 Analyze Selected",
                  command=self._analyze_selected).pack(side=tk.LEFT, padx=5)
        
        # === FILTERS ===
        filter_frame = ttk.Frame(main)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Search
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self._filter_list())
        ttk.Entry(filter_frame, textvariable=self.search_var, width=25).pack(side=tk.LEFT, padx=(5, 15))
        
        # Impact filter
        ttk.Label(filter_frame, text="Impact:").pack(side=tk.LEFT)
        self.impact_filter = tk.StringVar(value="All")
        impact_combo = ttk.Combobox(filter_frame, textvariable=self.impact_filter, width=15,
                                   values=["All", "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"])
        impact_combo.pack(side=tk.LEFT, padx=5)
        impact_combo.bind('<<ComboboxSelected>>', lambda e: self._filter_list())
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(filter_frame, textvariable=self.status_var, 
                 foreground=COLORS['text_muted']).pack(side=tk.RIGHT)
        
        # === MAIN CONTENT (Split Pane) ===
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Disclosures List
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        self._create_disclosures_list(left_frame)
        
        # Right: Details Panel
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        self._create_details_panel(right_frame)
    
    def _create_disclosures_list(self, parent):
        """Create the disclosures list."""
        frame = ttk.LabelFrame(parent, text="📄 Recent Disclosures")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        columns = ('company', 'title', 'date', 'impact')
        self.tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        
        self.tree.heading('company', text='Company')
        self.tree.heading('title', text='Title')
        self.tree.heading('date', text='Date')
        self.tree.heading('impact', text='Impact')
        
        self.tree.column('company', width=100, anchor='w')
        self.tree.column('title', width=250, anchor='w')
        self.tree.column('date', width=100, anchor='center')
        self.tree.column('impact', width=100, anchor='center')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Bind selection
        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<Double-1>', self._on_double_click)
        
        # Tags for impact colors
        self.tree.tag_configure('very_positive', foreground='#27ae60')
        self.tree.tag_configure('positive', foreground='#2ecc71')
        self.tree.tag_configure('neutral', foreground='#95a5a6')
        self.tree.tag_configure('negative', foreground='#e67e22')
        self.tree.tag_configure('very_negative', foreground='#e74c3c')
        self.tree.tag_configure('unanalyzed', foreground='#7f8c8d')
    
    def _create_details_panel(self, parent):
        """Create the details/analysis panel."""
        frame = ttk.LabelFrame(parent, text="🔍 Disclosure Analysis")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable content
        canvas = tk.Canvas(frame, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        
        self.details_frame = ttk.Frame(canvas)
        
        canvas.create_window((0, 0), window=self.details_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(canvas.find_withtag('all')[0], width=event.width - 20)
        
        self.details_frame.bind('<Configure>', on_configure)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Initial message
        ttk.Label(self.details_frame, text="Select a disclosure to view analysis",
                 font=get_font('body'), foreground=COLORS['text_muted']).pack(pady=50)
    
    def _refresh_disclosures(self):
        """Load disclosures from database."""
        if not self.scraper:
            self.status_var.set("⚠️ Scraper not available")
            return
        
        self.status_var.set("Loading disclosures...")
        
        try:
            self.disclosures = self.scraper.get_disclosures(limit=100)
            self._populate_list()
            self.status_var.set(f"Loaded {len(self.disclosures)} disclosures")
        except Exception as e:
            logger.error(f"Failed to load disclosures: {e}")
            self.status_var.set(f"Error: {e}")
    
    def _populate_list(self):
        """Populate the treeview with disclosures."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        search_text = self.search_var.get().lower()
        impact_filter = self.impact_filter.get()
        
        for disc in self.disclosures:
            # Apply filters
            if search_text:
                if (search_text not in (disc.get('company_symbol') or '').lower() and
                    search_text not in (disc.get('company_name') or '').lower() and
                    search_text not in (disc.get('title') or '').lower()):
                    continue
            
            impact_label = disc.get('impact_label', 'Unanalyzed')
            if impact_filter != "All" and impact_label != impact_filter:
                continue
            
            # Determine tag
            tag = impact_label.lower().replace(' ', '_') if impact_label else 'unanalyzed'
            
            # Impact display
            impact_display = impact_label if impact_label else '⏳ Pending'
            if disc.get('impact_score') is not None:
                score = disc['impact_score']
                icons = {2: '🚀', 1: '📈', 0: '➡️', -1: '📉', -2: '⚠️'}
                impact_display = f"{icons.get(score, '❓')} {impact_label}"
            
            self.tree.insert('', 'end', iid=disc['id'], values=(
                disc.get('company_symbol') or disc.get('company_name', 'N/A')[:10],
                (disc.get('title') or 'Untitled')[:50],
                disc.get('date_submitted', ''),
                impact_display
            ), tags=(tag,))
    
    def _filter_list(self):
        """Re-filter the list based on current filters."""
        self._populate_list()
    
    def _on_select(self, event):
        """Handle selection change."""
        selected = self.tree.selection()
        if not selected:
            return
        
        disc_id = int(selected[0])
        self.selected_disclosure = next(
            (d for d in self.disclosures if d['id'] == disc_id), None
        )
        
        if self.selected_disclosure:
            self._show_details(self.selected_disclosure)
    
    def _on_double_click(self, event):
        """Open PDF on double-click."""
        if self.selected_disclosure and self.selected_disclosure.get('pdf_url'):
            webbrowser.open(self.selected_disclosure['pdf_url'])
    
    def _show_details(self, disclosure: Dict):
        """Display disclosure details in panel."""
        # Clear existing content
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        # Company & Title
        ttk.Label(self.details_frame, 
                 text=disclosure.get('company_symbol') or disclosure.get('company_name', 'Unknown'),
                 font=('Helvetica', 16, 'bold'),
                 foreground=COLORS['primary']).pack(anchor='w', padx=10, pady=(10, 0))
        
        ttk.Label(self.details_frame,
                 text=disclosure.get('title', 'Untitled'),
                 font=get_font('body'),
                 foreground=COLORS['text_primary'],
                 wraplength=400).pack(anchor='w', padx=10, pady=(5, 10))
        
        # Date & PDF Link
        info_frame = ttk.Frame(self.details_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text=f"📅 {disclosure.get('date_submitted', 'N/A')}",
                 foreground=COLORS['text_muted']).pack(side=tk.LEFT)
        
        if disclosure.get('pdf_url'):
            ttk.Button(info_frame, text="📄 Open PDF",
                      command=lambda: webbrowser.open(disclosure['pdf_url'])).pack(side=tk.RIGHT)
        
        ttk.Separator(self.details_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        # AI Analysis
        if disclosure.get('ai_summary'):
            # Impact Badge
            impact_score = disclosure.get('impact_score', 0)
            impact_label = disclosure.get('impact_label', 'Neutral')
            icons = {2: '🚀', 1: '📈', 0: '➡️', -1: '📉', -2: '⚠️'}
            colors = {2: '#27ae60', 1: '#2ecc71', 0: '#95a5a6', -1: '#e67e22', -2: '#e74c3c'}
            
            impact_frame = ttk.Frame(self.details_frame)
            impact_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(impact_frame, text=f"{icons.get(impact_score, '❓')} Expected Impact: ",
                     font=get_font('subheading')).pack(side=tk.LEFT)
            ttk.Label(impact_frame, text=impact_label,
                     font=get_font('subheading'),
                     foreground=colors.get(impact_score, '#7f8c8d')).pack(side=tk.LEFT)
            
            # Summary
            ttk.Label(self.details_frame, text="📝 AI Summary",
                     font=get_font('subheading'),
                     foreground=COLORS['text_primary']).pack(anchor='w', padx=10, pady=(10, 5))
            
            summary_text = tk.Text(self.details_frame, height=6, wrap=tk.WORD,
                                  bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                                  font=get_font('body'), relief='flat')
            summary_text.insert('1.0', disclosure.get('ai_summary', 'No summary available'))
            summary_text.config(state='disabled')
            summary_text.pack(fill=tk.X, padx=10, pady=5)
            
            # Highlights
            highlights = disclosure.get('key_highlights')
            if highlights:
                try:
                    import json
                    if isinstance(highlights, str):
                        highlights = json.loads(highlights)
                    
                    if highlights:
                        ttk.Label(self.details_frame, text="⭐ Key Highlights",
                                 font=get_font('subheading'),
                                 foreground=COLORS['text_primary']).pack(anchor='w', padx=10, pady=(10, 5))
                        
                        for hl in highlights[:5]:
                            ttk.Label(self.details_frame, text=f"  • {hl}",
                                     foreground=COLORS['text_secondary'],
                                     wraplength=380).pack(anchor='w', padx=15)
                except:
                    pass
        else:
            # Not analyzed yet
            ttk.Label(self.details_frame, text="⏳ Not yet analyzed",
                     font=get_font('subheading'),
                     foreground=COLORS['text_muted']).pack(pady=20)
            
            ttk.Button(self.details_frame, text="🤖 Analyze Now",
                      command=self._analyze_selected).pack(pady=10)
    
    def _sync_disclosures(self):
        """Sync disclosures from NGX website."""
        if not self.scraper:
            messagebox.showerror("Error", "Disclosure scraper not available")
            return
        
        self.status_var.set("Syncing from NGX...")
        
        def sync():
            try:
                count = self.scraper.sync(limit=100)
                self.frame.after(0, lambda: [
                    self.status_var.set(f"Synced {count} new disclosures"),
                    self._refresh_disclosures()
                ])
            except Exception as e:
                logger.error(f"Sync failed: {e}")
                self.frame.after(0, lambda: self.status_var.set(f"Sync failed: {e}"))
        
        threading.Thread(target=sync, daemon=True).start()
    
    def _analyze_selected(self):
        """Analyze the selected disclosure."""
        if not self.selected_disclosure:
            messagebox.showinfo("Info", "Please select a disclosure first")
            return
        
        if not self.pdf_parser or not self.pdf_parser.available:
            messagebox.showerror("Error", "PDF parser not available. Install pymupdf.")
            return
        
        if not self.analyzer:
            messagebox.showerror("Error", "Disclosure analyzer not available")
            return
        
        disc = self.selected_disclosure
        self.status_var.set(f"Analyzing {disc.get('company_symbol', 'disclosure')}...")
        
        def analyze():
            try:
                # Download and extract PDF
                pdf_url = disc.get('pdf_url')
                if not pdf_url:
                    raise ValueError("No PDF URL available")
                
                pdf_text = self.pdf_parser.extract_from_url(pdf_url)
                if not pdf_text:
                    raise ValueError("Could not extract text from PDF")
                
                pdf_text = self.pdf_parser.clean_text(pdf_text)
                
                # Analyze
                result = self.analyzer.analyze(disc, pdf_text)
                
                # Store results
                import json
                self.scraper.update_disclosure(
                    disc['id'],
                    pdf_text=pdf_text[:10000],  # Truncate for storage
                    ai_summary=result.get('summary', ''),
                    impact_score=result.get('impact_score', 0),
                    impact_label=result.get('impact_label', 'Neutral'),
                    key_highlights=json.dumps(result.get('highlights', [])),
                    processed_at=datetime.now().isoformat()
                )
                
                # Update UI
                self.frame.after(0, lambda: [
                    self.status_var.set("Analysis complete"),
                    self._refresh_disclosures(),
                    messagebox.showinfo("Success", f"Analyzed: {disc.get('title', 'disclosure')[:50]}")
                ])
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.frame.after(0, lambda: [
                    self.status_var.set(f"Analysis failed: {e}"),
                    messagebox.showerror("Error", f"Analysis failed: {e}")
                ])
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def refresh(self):
        """Public refresh method."""
        self._refresh_disclosures()
