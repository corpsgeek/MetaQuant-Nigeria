"""
Report Generator for MetaQuant Nigeria.
Generates CSV and PDF reports for portfolio, predictions, and analysis.
"""

import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import io

logger = logging.getLogger(__name__)

# Try to import PDF library
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF export disabled. Install with: pip install reportlab")


class ReportGenerator:
    """
    Generates reports in CSV and PDF formats.
    
    Supports:
    - Portfolio holdings export
    - ML predictions export
    - Screener results export
    - Full portfolio report (PDF)
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.home() / "Documents" / "MetaQuant_Reports"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_available = REPORTLAB_AVAILABLE
    
    def export_csv(self, data: List[Dict], filename: str, columns: Optional[List[str]] = None) -> str:
        """
        Export data to CSV file.
        
        Args:
            data: List of dictionaries to export
            filename: Output filename (without extension)
            columns: Optional list of columns to include (default: all)
            
        Returns:
            Path to generated file
        """
        if not data:
            raise ValueError("No data to export")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{filename}_{timestamp}.csv"
        
        # Determine columns
        if columns is None:
            columns = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"CSV exported to {filepath}")
        return str(filepath)
    
    def export_portfolio_csv(self, holdings: List[Dict]) -> str:
        """Export portfolio holdings to CSV."""
        columns = ['symbol', 'name', 'quantity', 'avg_cost', 'current_price', 
                  'market_value', 'pnl', 'pnl_pct', 'sector', 'weight']
        return self.export_csv(holdings, "portfolio_holdings", columns)
    
    def export_predictions_csv(self, predictions: List[Dict]) -> str:
        """Export ML predictions to CSV."""
        columns = ['symbol', 'name', 'sector', 'signal', 'score', 
                  'confidence', 'price', 'target']
        return self.export_csv(predictions, "ml_predictions", columns)
    
    def export_screener_csv(self, results: List[Dict]) -> str:
        """Export screener results to CSV."""
        columns = ['symbol', 'name', 'sector', 'price', 'change_1d', 
                  'change_1w', 'volume', 'pe', 'dividend_yield']
        return self.export_csv(results, "screener_results", columns)
    
    def generate_portfolio_pdf(
        self,
        portfolio_name: str,
        summary: Dict[str, Any],
        holdings: List[Dict],
        risk_metrics: Optional[Dict] = None,
        predictions: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a comprehensive portfolio PDF report.
        
        Args:
            portfolio_name: Name of the portfolio
            summary: Portfolio summary (value, pnl, etc.)
            holdings: List of holdings
            risk_metrics: Optional risk metrics
            predictions: Optional ML predictions
            
        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab not installed. Run: pip install reportlab")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"portfolio_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor('#2C3E50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#34495E')
        )
        
        # Title
        elements.append(Paragraph(f"📊 {portfolio_name} Report", title_style))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        # Portfolio Summary
        elements.append(Paragraph("Portfolio Summary", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Value', f"₦{summary.get('total_value', 0):,.2f}"],
            ['Total Cost', f"₦{summary.get('total_cost', 0):,.2f}"],
            ['Unrealized P&L', f"₦{summary.get('unrealized_pnl', 0):,.2f}"],
            ['Return %', f"{summary.get('return_pct', 0):.2f}%"],
            ['Holdings', str(summary.get('holdings_count', 0))],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Holdings Table
        if holdings:
            elements.append(Paragraph("Holdings", heading_style))
            
            holdings_data = [['Symbol', 'Qty', 'Avg Cost', 'Price', 'Value', 'P&L %']]
            for h in holdings[:20]:  # Limit to 20 for readability
                pnl_pct = h.get('pnl_pct', 0)
                holdings_data.append([
                    h.get('symbol', ''),
                    str(h.get('quantity', 0)),
                    f"₦{h.get('avg_cost', 0):,.2f}",
                    f"₦{h.get('current_price', 0):,.2f}",
                    f"₦{h.get('market_value', 0):,.2f}",
                    f"{pnl_pct:+.2f}%"
                ])
            
            holdings_table = Table(holdings_data, colWidths=[1*inch, 0.6*inch, 1.1*inch, 1.1*inch, 1.2*inch, 0.8*inch])
            holdings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            elements.append(holdings_table)
            elements.append(Spacer(1, 20))
        
        # Risk Metrics
        if risk_metrics:
            elements.append(Paragraph("Risk Metrics", heading_style))
            
            risk_data = [
                ['Metric', 'Value'],
                ['Beta', f"{risk_metrics.get('beta', 0):.2f}"],
                ['Volatility', f"{risk_metrics.get('volatility', 0):.1f}%"],
                ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.2f}"],
                ['Max Drawdown', f"{risk_metrics.get('max_drawdown', 0):.1f}%"],
                ['VaR (95%)', f"{risk_metrics.get('var_95', 0):.1f}%"],
            ]
            
            risk_table = Table(risk_data, colWidths=[2*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FDEBD0')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 20))
        
        # ML Predictions
        if predictions:
            elements.append(Paragraph("ML Signal Summary", heading_style))
            
            buy_count = sum(1 for p in predictions if p.get('signal') == 'BUY')
            sell_count = sum(1 for p in predictions if p.get('signal') == 'SELL')
            hold_count = sum(1 for p in predictions if p.get('signal') == 'HOLD')
            
            pred_summary = [
                ['Signal', 'Count'],
                ['🟢 BUY', str(buy_count)],
                ['🔴 SELL', str(sell_count)],
                ['🟡 HOLD', str(hold_count)],
            ]
            
            pred_table = Table(pred_summary, colWidths=[2*inch, 2*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
            ]))
            elements.append(pred_table)
        
        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "Generated by MetaQuant Nigeria | www.metaquant.ng",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray, alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    def get_output_dir(self) -> str:
        """Return the output directory path."""
        return str(self.output_dir)


# Convenience functions
def export_to_csv(data: List[Dict], filename: str) -> str:
    """Quick export data to CSV."""
    generator = ReportGenerator()
    return generator.export_csv(data, filename)


def generate_portfolio_report(portfolio_name: str, summary: Dict, holdings: List[Dict], **kwargs) -> str:
    """Quick generate portfolio PDF report."""
    generator = ReportGenerator()
    return generator.generate_portfolio_pdf(portfolio_name, summary, holdings, **kwargs)
