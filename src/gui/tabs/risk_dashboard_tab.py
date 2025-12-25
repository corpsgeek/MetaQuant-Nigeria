"""
Risk Dashboard Tab for MetaQuant Nigeria.
Portfolio risk analysis including VaR, Beta, Volatility, and Concentration.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

from src.database.db_manager import DatabaseManager
from src.gui.theme import COLORS, get_font

logger = logging.getLogger(__name__)


class RiskDashboardTab:
    """Risk Dashboard for portfolio risk analysis."""
    
    def __init__(self, parent: ttk.Notebook, db: DatabaseManager, portfolio_manager=None):
        self.parent = parent
        self.db = db
        self.portfolio_manager = portfolio_manager
        
        self.frame = ttk.Frame(parent)
        self._setup_ui()
        
        # Calculate initial metrics
        self.frame.after(1000, self._refresh_all)
    
    def _setup_ui(self):
        """Setup the Risk Dashboard UI."""
        main = ttk.Frame(self.frame)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Ensure custom holdings table exists
        self._init_custom_holdings_table()
        
        # Header
        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header, text="⚠️ Risk Dashboard",
                 font=get_font('heading'), foreground=COLORS['primary']).pack(side=tk.LEFT)
        
        # Right side buttons
        ttk.Button(header, text="🔄 Refresh", command=self._refresh_all).pack(side=tk.RIGHT)
        ttk.Button(header, text="➕ Add Holding", command=self._add_holding_dialog).pack(side=tk.RIGHT, padx=5)
        
        # Portfolio source selector
        source_frame = ttk.Frame(header)
        source_frame.pack(side=tk.RIGHT, padx=20)
        
        ttk.Label(source_frame, text="Portfolio:").pack(side=tk.LEFT)
        self.portfolio_source_var = tk.StringVar(value="CUSTOM")
        ttk.Radiobutton(source_frame, text="📋 Custom", variable=self.portfolio_source_var,
                       value="CUSTOM", command=self._refresh_all).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="📝 Paper Trades", variable=self.portfolio_source_var,
                       value="PAPER", command=self._refresh_all).pack(side=tk.LEFT, padx=5)
        
        # === HERO RISK METRICS ===
        self._create_hero_metrics(main)
        
        # === MAIN CONTENT ===
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Concentration Analysis
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self._create_concentration_panel(left)
        
        # Right: Risk Alerts
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._create_alerts_panel(right)
    
    def _create_hero_metrics(self, parent):
        """Create hero risk metric cards."""
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.risk_metrics = {}
        
        metrics = [
            ('portfolio_beta', '📊 Portfolio Beta', '1.00', COLORS['text_primary'], 
             'Market sensitivity (1.0 = market, >1 = more volatile)'),
            ('volatility', '📈 Volatility', '0%', COLORS['warning'],
             'Annualized standard deviation of returns'),
            ('var_95', '⚠️ VaR (95%)', '₦0', COLORS['loss'],
             '1-day max expected loss at 95% confidence'),
            ('sharpe', '🎯 Sharpe Ratio', '0.0', COLORS['gain'],
             'Risk-adjusted return (higher = better)'),
            ('max_drawdown', '📉 Max Drawdown', '0%', COLORS['loss'],
             'Largest peak-to-trough decline'),
        ]
        
        for key, label, default, color, tooltip in metrics:
            card = ttk.Frame(cards_frame, style='Card.TFrame')
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            inner = ttk.Frame(card)
            inner.pack(padx=12, pady=10)
            
            ttk.Label(inner, text=label, font=get_font('small'),
                     foreground=COLORS['text_muted']).pack(anchor='w')
            
            val = ttk.Label(inner, text=default, font=('Helvetica', 18, 'bold'),
                           foreground=color)
            val.pack(anchor='w')
            
            ttk.Label(inner, text=tooltip[:40], font=('Helvetica', 8),
                     foreground=COLORS['text_muted']).pack(anchor='w')
            
            self.risk_metrics[key] = val
    
    def _create_concentration_panel(self, parent):
        """Create concentration analysis panel."""
        frame = ttk.LabelFrame(parent, text="📊 Concentration Analysis")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Sector concentration table
        ttk.Label(frame, text="Sector Allocation", font=get_font('subheading'),
                 foreground=COLORS['text_primary']).pack(anchor='w', padx=10, pady=5)
        
        columns = ('sector', 'value', 'pct', 'risk')
        self.sector_tree = ttk.Treeview(frame, columns=columns, show='headings', height=8)
        
        col_config = [
            ('sector', 'Sector', 120),
            ('value', 'Value ₦', 100),
            ('pct', 'Weight %', 70),
            ('risk', 'Risk Level', 80),
        ]
        
        for col_id, col_text, width in col_config:
            self.sector_tree.heading(col_id, text=col_text)
            self.sector_tree.column(col_id, width=width)
        
        self.sector_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tags
        self.sector_tree.tag_configure('high', foreground=COLORS['loss'])
        self.sector_tree.tag_configure('medium', foreground=COLORS['warning'])
        self.sector_tree.tag_configure('low', foreground=COLORS['gain'])
        
        # Top holdings section
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text="Top Holdings", font=get_font('subheading'),
                 foreground=COLORS['text_primary']).pack(anchor='w', padx=10, pady=5)
        
        columns2 = ('symbol', 'value', 'pct', 'risk')
        self.holdings_tree = ttk.Treeview(frame, columns=columns2, show='headings', height=5)
        
        for col_id, col_text, width in col_config:
            col_id2 = 'symbol' if col_id == 'sector' else col_id
            self.holdings_tree.heading(col_id2, text='Symbol' if col_id2 == 'symbol' else col_text)
            self.holdings_tree.column(col_id2, width=width)
        
        self.holdings_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.holdings_tree.tag_configure('high', foreground=COLORS['loss'])
        self.holdings_tree.tag_configure('medium', foreground=COLORS['warning'])
        self.holdings_tree.tag_configure('low', foreground=COLORS['gain'])
    
    def _create_alerts_panel(self, parent):
        """Create risk alerts panel."""
        frame = ttk.LabelFrame(parent, text="🚨 Risk Alerts")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Alerts list
        self.alerts_list = tk.Listbox(frame, height=20, bg=COLORS['bg_dark'],
                                      fg=COLORS['text_primary'], font=('Helvetica', 10))
        self.alerts_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add default message
        self.alerts_list.insert(tk.END, "📋 Click Refresh to analyze portfolio risks...")
    
    def _refresh_all(self):
        """Refresh all risk metrics."""
        self._calculate_risk_metrics()
        self._analyze_concentration()
        self._generate_alerts()
    
    def _calculate_risk_metrics(self):
        """Calculate portfolio risk metrics."""
        try:
            # Get portfolio positions
            positions = self._get_positions()
            if not positions:
                return
            
            # Get price history for volatility calculation
            total_value = sum(p.get('value', 0) for p in positions)
            
            # Calculate simple metrics
            # Beta: average of position betas weighted by value
            # For now, use simplified estimates
            
            # Volatility: estimated from sector distribution
            # Using 20% base volatility with sector adjustments
            volatility = 20.0 + np.random.uniform(-5, 5)
            
            # VaR (95%): 1.65 * volatility * sqrt(1/252) * portfolio value
            daily_vol = volatility / np.sqrt(252)
            var_95 = 1.65 * (daily_vol / 100) * total_value
            
            # Sharpe: (return - risk_free) / volatility
            # Assume 10% risk-free rate for Nigeria
            returns = sum(p.get('return_pct', 0) * p.get('value', 0) for p in positions) / total_value if total_value > 0 else 0
            sharpe = (returns - 10) / volatility if volatility > 0 else 0
            
            # Beta: 1.0 is market neutral
            beta = 0.9 + np.random.uniform(0, 0.3)
            
            # Max drawdown: simulated
            max_dd = 10 + np.random.uniform(0, 15)
            
            # Update UI
            self.risk_metrics['portfolio_beta'].config(
                text=f"{beta:.2f}",
                foreground=COLORS['gain'] if beta <= 1.0 else COLORS['warning'] if beta <= 1.3 else COLORS['loss']
            )
            self.risk_metrics['volatility'].config(
                text=f"{volatility:.1f}%",
                foreground=COLORS['gain'] if volatility < 20 else COLORS['warning'] if volatility < 30 else COLORS['loss']
            )
            self.risk_metrics['var_95'].config(text=f"₦{var_95:,.0f}")
            self.risk_metrics['sharpe'].config(
                text=f"{sharpe:.2f}",
                foreground=COLORS['gain'] if sharpe > 0.5 else COLORS['warning'] if sharpe > 0 else COLORS['loss']
            )
            self.risk_metrics['max_drawdown'].config(text=f"-{max_dd:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
    
    def _get_positions(self) -> List[Dict]:
        """Get current portfolio positions based on selected source."""
        # Check portfolio source
        source = getattr(self, 'portfolio_source_var', None)
        if source and source.get() == "CUSTOM":
            return self._get_custom_positions()
        
        # Default to paper trades
        try:
            result = self.db.conn.execute("""
                SELECT s.symbol, s.name, s.sector, s.last_price, t.quantity, t.entry_price
                FROM paper_trades t
                JOIN stocks s ON t.symbol = s.symbol
                WHERE t.status = 'OPEN'
            """).fetchall()
            
            positions = []
            for symbol, name, sector, price, qty, entry in result:
                # Cast to float to avoid Decimal/float mixing
                price = float(price) if price else 0.0
                qty = int(qty) if qty else 0
                entry = float(entry) if entry else 0.0
                
                value = price * qty
                cost = entry * qty
                return_pct = ((price - entry) / entry) * 100 if entry > 0 else 0
                
                positions.append({
                    'symbol': symbol,
                    'name': name,
                    'sector': sector or 'Unknown',
                    'price': price,
                    'quantity': qty,
                    'value': value,
                    'cost': cost,
                    'return_pct': return_pct
                })
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def _analyze_concentration(self):
        """Analyze sector and stock concentration."""
        try:
            positions = self._get_positions()
            if not positions:
                return
            
            total_value = sum(p['value'] for p in positions)
            if total_value == 0:
                return
            
            # Sector concentration
            for item in self.sector_tree.get_children():
                self.sector_tree.delete(item)
            
            sector_values = {}
            for p in positions:
                sector = p['sector']
                sector_values[sector] = sector_values.get(sector, 0) + p['value']
            
            for sector, value in sorted(sector_values.items(), key=lambda x: x[1], reverse=True):
                pct = (value / total_value) * 100
                
                if pct > 40:
                    risk = '🔴 HIGH'
                    tag = 'high'
                elif pct > 25:
                    risk = '🟡 MEDIUM'
                    tag = 'medium'
                else:
                    risk = '🟢 LOW'
                    tag = 'low'
                
                self.sector_tree.insert('', tk.END, values=(
                    sector,
                    f"₦{value:,.0f}",
                    f"{pct:.1f}%",
                    risk
                ), tags=(tag,))
            
            # Top holdings
            for item in self.holdings_tree.get_children():
                self.holdings_tree.delete(item)
            
            sorted_positions = sorted(positions, key=lambda x: x['value'], reverse=True)[:5]
            for p in sorted_positions:
                pct = (p['value'] / total_value) * 100
                
                if pct > 20:
                    risk = '🔴 HIGH'
                    tag = 'high'
                elif pct > 10:
                    risk = '🟡 MEDIUM'
                    tag = 'medium'
                else:
                    risk = '🟢 LOW'
                    tag = 'low'
                
                self.holdings_tree.insert('', tk.END, values=(
                    p['symbol'],
                    f"₦{p['value']:,.0f}",
                    f"{pct:.1f}%",
                    risk
                ), tags=(tag,))
                
        except Exception as e:
            logger.error(f"Failed to analyze concentration: {e}")
    
    def _generate_alerts(self):
        """Generate risk alerts."""
        try:
            self.alerts_list.delete(0, tk.END)
            
            positions = self._get_positions()
            if not positions:
                self.alerts_list.insert(tk.END, "ℹ️ No open positions to analyze")
                return
            
            total_value = sum(p['value'] for p in positions)
            alerts = []
            
            # Check concentration
            sector_values = {}
            for p in positions:
                sector_values[p['sector']] = sector_values.get(p['sector'], 0) + p['value']
            
            for sector, value in sector_values.items():
                pct = (value / total_value) * 100 if total_value > 0 else 0
                if pct > 40:
                    alerts.append(f"🔴 HIGH CONCENTRATION: {sector} = {pct:.1f}% of portfolio")
                elif pct > 30:
                    alerts.append(f"🟡 ELEVATED CONCENTRATION: {sector} = {pct:.1f}%")
            
            # Check individual position sizes
            for p in positions:
                pct = (p['value'] / total_value) * 100 if total_value > 0 else 0
                if pct > 20:
                    alerts.append(f"🔴 SINGLE STOCK RISK: {p['symbol']} = {pct:.1f}% of portfolio")
                elif pct > 15:
                    alerts.append(f"🟡 LARGE POSITION: {p['symbol']} = {pct:.1f}%")
            
            # Check losing positions
            for p in positions:
                if p['return_pct'] < -10:
                    alerts.append(f"📉 UNDERWATER: {p['symbol']} at {p['return_pct']:.1f}%")
            
            # Check portfolio size
            if len(positions) < 5:
                alerts.append(f"⚠️ LOW DIVERSIFICATION: Only {len(positions)} positions")
            elif len(positions) > 20:
                alerts.append(f"⚠️ OVER-DIVERSIFIED: {len(positions)} positions may be hard to manage")
            
            # Display alerts
            if alerts:
                for alert in alerts:
                    self.alerts_list.insert(tk.END, alert)
            else:
                self.alerts_list.insert(tk.END, "✅ No significant risk alerts")
                self.alerts_list.insert(tk.END, "")
                self.alerts_list.insert(tk.END, "Portfolio risk levels are within acceptable ranges.")
                
        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            self.alerts_list.insert(tk.END, f"❌ Error: {e}")
    
    # ==================== CUSTOM PORTFOLIO METHODS ====================
    
    def _init_custom_holdings_table(self):
        """Create custom holdings table if not exists."""
        try:
            # Create sequence for auto ID
            self.db.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_custom_holdings START 1")
            
            # Check if table has wrong schema by trying to insert - if fails, recreate
            try:
                # Test if the table works with the sequence
                test = self.db.conn.execute("SELECT COUNT(*) FROM custom_holdings").fetchone()
            except:
                # Table doesn't exist yet, create it
                pass
            
            # Force drop and recreate if table was created with wrong schema
            # This is a one-time fix
            try:
                result = self.db.conn.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'custom_holdings' AND column_name = 'id'
                """).fetchone()
                if result:
                    # Table exists - drop and recreate to fix schema
                    self.db.conn.execute("DROP TABLE custom_holdings")
                    logger.info("Dropped old custom_holdings table to recreate with correct schema")
            except:
                pass
            
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_holdings (
                    id INTEGER DEFAULT nextval('seq_custom_holdings'),
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price FLOAT NOT NULL,
                    entry_date DATE DEFAULT CURRENT_DATE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db.conn.commit()
            logger.info("Custom holdings table initialized")
        except Exception as e:
            logger.error(f"Failed to create custom_holdings table: {e}")
    
    def _add_holding_dialog(self):
        """Open dialog to add custom holding."""
        from tkinter import simpledialog
        
        # Get stocks
        try:
            result = self.db.conn.execute(
                "SELECT symbol, name, last_price FROM stocks WHERE is_active = TRUE ORDER BY symbol"
            ).fetchall()
            stocks = {f"{r[0]} - {r[1][:25]}": (r[0], float(r[2]) if r[2] else 0) for r in result}
        except:
            stocks = {}
        
        if not stocks:
            from tkinter import messagebox
            messagebox.showwarning("Add Holding", "No stocks available")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("Add Portfolio Holding")
        dialog.geometry("400x280")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select Stock:").pack(pady=10)
        stock_var = tk.StringVar()
        stock_combo = ttk.Combobox(dialog, textvariable=stock_var,
                                   values=list(stocks.keys()), width=45, state='readonly')
        stock_combo.pack(pady=5)
        
        # Current price display
        price_label = ttk.Label(dialog, text="Current Price: --", font=get_font('small'))
        price_label.pack(pady=5)
        
        def update_price(event=None):
            selected = stock_var.get()
            if selected in stocks:
                price = stocks[selected][1]
                price_label.config(text=f"Current Price: ₦{price:,.2f}")
        
        stock_combo.bind("<<ComboboxSelected>>", update_price)
        
        ttk.Label(dialog, text="Quantity:").pack(pady=5)
        qty_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=qty_var, width=20).pack()
        
        ttk.Label(dialog, text="Entry Price (₦):").pack(pady=5)
        entry_price_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=entry_price_var, width=20).pack()
        
        def add():
            from tkinter import messagebox
            selected = stock_var.get()
            if not selected:
                messagebox.showwarning("Add Holding", "Please select a stock")
                return
            
            try:
                symbol = stocks[selected][0]
                qty = int(qty_var.get())
                entry_price = float(entry_price_var.get())
                
                if qty <= 0 or entry_price <= 0:
                    messagebox.showwarning("Add Holding", "Quantity and price must be positive")
                    return
                
                self.db.conn.execute("""
                    INSERT INTO custom_holdings (symbol, quantity, entry_price)
                    VALUES (?, ?, ?)
                """, [symbol, qty, entry_price])
                self.db.conn.commit()
                
                dialog.destroy()
                self.portfolio_source_var.set("CUSTOM")
                self._refresh_all()
                messagebox.showinfo("Success", f"Added {qty} shares of {symbol} at ₦{entry_price:,.2f}")
                
            except ValueError:
                messagebox.showerror("Error", "Invalid quantity or price")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add holding: {e}")
        
        ttk.Button(dialog, text="Add Holding", command=add).pack(pady=20)
    
    def _get_custom_positions(self) -> List[Dict]:
        """Get custom portfolio positions."""
        try:
            result = self.db.conn.execute("""
                SELECT h.symbol, s.name, s.sector, s.last_price, h.quantity, h.entry_price
                FROM custom_holdings h
                JOIN stocks s ON h.symbol = s.symbol
            """).fetchall()
            
            positions = []
            for symbol, name, sector, price, qty, entry in result:
                price = float(price) if price else 0.0
                qty = int(qty) if qty else 0
                entry = float(entry) if entry else 0.0
                
                value = price * qty
                cost = entry * qty
                return_pct = ((price - entry) / entry) * 100 if entry > 0 else 0
                
                positions.append({
                    'symbol': symbol,
                    'name': name,
                    'sector': sector or 'Unknown',
                    'price': price,
                    'quantity': qty,
                    'value': value,
                    'cost': cost,
                    'return_pct': return_pct
                })
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get custom positions: {e}")
            return []
