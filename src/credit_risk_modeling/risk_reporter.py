"""Risk Reporting: Generate executive dashboards and KPI reports"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from credit_risk_modeling.config import FIGURES_DIR, PROCESSED_DATA_DIR


class RiskReporter:
    """Generate executive-ready risk reports and dashboards."""
    
    def __init__(self):
        logger.info("✓ RiskReporter initialized")
    
    def generate_kpi_summary(self, scoring_results_df: pd.DataFrame, 
                            monitoring_metrics=None) -> dict:
        """
        Generate key performance indicators.
        
        Returns:
            dict: Portfolio KPIs
        """
        df = scoring_results_df.copy()
        
        kpis = {
            'portfolio_size': len(df),
            'total_exposure': df['ead'].sum(),
            'total_expected_loss': df['expected_loss'].sum(),
            'expected_loss_pct': (df['expected_loss'].sum() / df['ead'].sum() * 100) if df['ead'].sum() > 0 else 0,
            
            'approval_rate': (df['decision'] == 'APPROVE').sum() / len(df),
            'manual_review_rate': (df['decision'] == 'MANUAL_REVIEW').sum() / len(df),
            'denial_rate': (df['decision'] == 'DENY').sum() / len(df),
            
            'avg_pd': df['pd'].mean(),
            'median_pd': df['pd'].median(),
            'max_pd': df['pd'].max(),
            
            'avg_confidence': df['confidence'].mean(),
            'avg_risk_score': df['risk_score'].mean(),
            
            'low_risk_count': (df['risk_tier'] == 'LOW').sum(),
            'medium_risk_count': (df['risk_tier'] == 'MEDIUM').sum(),
            'high_risk_count': (df['risk_tier'] == 'HIGH').sum(),
            
            'approved_exposure': df[df['decision'] == 'APPROVE']['ead'].sum(),
            'denied_exposure': df[df['decision'] == 'DENY']['ead'].sum(),
        }
        
        return kpis
    
    def generate_html_report(self, scoring_results_df: pd.DataFrame, 
                            output_path: Path = None) -> str:
        """
        Generate HTML dashboard report.
        
        Args:
            scoring_results_df: Batch scoring results
            output_path: Where to save HTML file
        
        Returns:
            str: HTML content
        """
        kpis = self.generate_kpi_summary(scoring_results_df)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Credit Risk Portfolio Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .kpi-container {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .kpi-card {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .kpi-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .kpi-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px 0; }}
        .kpi-unit {{ font-size: 12px; color: #95a5a6; }}
        .section {{ background-color: white; padding: 20px; margin: 15px 0; border-radius: 5px; }}
        .section-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .risk-low {{ background-color: #d5f4e6; color: #27ae60; }}
        .risk-medium {{ background-color: #ffeaa7; color: #f39c12; }}
        .risk-high {{ background-color: #fab1a0; color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th {{ background-color: #ecf0f1; padding: 10px; text-align: left; font-weight: bold; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .footer {{ text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Credit Risk Portfolio Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Portfolio Size</div>
            <div class="kpi-value">{kpis['portfolio_size']:,}</div>
            <div class="kpi-unit">applicants</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Total Exposure</div>
            <div class="kpi-value">${kpis['total_exposure']:,.0f}</div>
            <div class="kpi-unit">EAD</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Expected Loss</div>
            <div class="kpi-value">${kpis['total_expected_loss']:,.0f}</div>
            <div class="kpi-unit">{kpis['expected_loss_pct']:.2f}% of EAD</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Approval Rate</div>
            <div class="kpi-value">{kpis['approval_rate']*100:.1f}%</div>
            <div class="kpi-unit">approved</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Manual Review Rate</div>
            <div class="kpi-value">{kpis['manual_review_rate']*100:.1f}%</div>
            <div class="kpi-unit">under review</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Denial Rate</div>
            <div class="kpi-value">{kpis['denial_rate']*100:.1f}%</div>
            <div class="kpi-unit">denied</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Average PD</div>
            <div class="kpi-value">{kpis['avg_pd']:.2%}</div>
            <div class="kpi-unit">default risk</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Model Confidence</div>
            <div class="kpi-value">{kpis['avg_confidence']:.2%}</div>
            <div class="kpi-unit">average</div>
        </div>
        
        <div class="kpi-card">
            <div class="kpi-label">Avg Risk Score</div>
            <div class="kpi-value">{kpis['avg_risk_score']:.0f}/100</div>
            <div class="kpi-unit">portfolio</div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-title">Risk Distribution</div>
        <table>
            <tr>
                <th>Risk Tier</th>
                <th>Count</th>
                <th>% of Portfolio</th>
                <th>Avg PD</th>
            </tr>
            <tr class="risk-low">
                <td>LOW</td>
                <td>{kpis['low_risk_count']:,}</td>
                <td>{kpis['low_risk_count']/kpis['portfolio_size']*100:.1f}%</td>
                <td>{scoring_results_df[scoring_results_df['risk_tier']=='LOW']['pd'].mean():.2%}</td>
            </tr>
            <tr class="risk-medium">
                <td>MEDIUM</td>
                <td>{kpis['medium_risk_count']:,}</td>
                <td>{kpis['medium_risk_count']/kpis['portfolio_size']*100:.1f}%</td>
                <td>{scoring_results_df[scoring_results_df['risk_tier']=='MEDIUM']['pd'].mean():.2%}</td>
            </tr>
            <tr class="risk-high">
                <td>HIGH</td>
                <td>{kpis['high_risk_count']:,}</td>
                <td>{kpis['high_risk_count']/kpis['portfolio_size']*100:.1f}%</td>
                <td>{scoring_results_df[scoring_results_df['risk_tier']=='HIGH']['pd'].mean():.2%}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <div class="section-title">Decision Breakdown</div>
        <table>
            <tr>
                <th>Decision</th>
                <th>Count</th>
                <th>Exposure ($)</th>
                <th>% of Portfolio</th>
            </tr>
            <tr>
                <td>APPROVE</td>
                <td>{(scoring_results_df['decision']=='APPROVE').sum():,}</td>
                <td>${kpis['approved_exposure']:,.0f}</td>
                <td>{kpis['approved_exposure']/kpis['total_exposure']*100:.1f}%</td>
            </tr>
            <tr>
                <td>MANUAL REVIEW</td>
                <td>{(scoring_results_df['decision']=='MANUAL_REVIEW').sum():,}</td>
                <td>${scoring_results_df[scoring_results_df['decision']=='MANUAL_REVIEW']['ead'].sum():,.0f}</td>
                <td>{scoring_results_df[scoring_results_df['decision']=='MANUAL_REVIEW']['ead'].sum()/kpis['total_exposure']*100:.1f}%</td>
            </tr>
            <tr>
                <td>DENY</td>
                <td>{(scoring_results_df['decision']=='DENY').sum():,}</td>
                <td>${kpis['denied_exposure']:,.0f}</td>
                <td>{kpis['denied_exposure']/kpis['total_exposure']*100:.1f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="footer">
        <p>This report was automatically generated by Credit Risk Modeling System</p>
        <p>For questions, contact: risk.analytics@company.com</p>
    </div>
</body>
</html>
"""
        
        if output_path is None:
            output_path = FIGURES_DIR / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.success(f"✓ HTML report saved to {output_path}")
        return html
    
    def export_metrics_csv(self, scoring_results_df: pd.DataFrame, 
                          output_path: Path = None) -> pd.DataFrame:
        """Export summary metrics to CSV."""
        kpis = self.generate_kpi_summary(scoring_results_df)
        
        metrics_df = pd.DataFrame([kpis])
        metrics_df.insert(0, 'report_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / f"kpi_summary_{datetime.now().strftime('%Y%m%d')}.csv"
        
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"✓ Metrics exported to {output_path}")
        
        return metrics_df


if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def generate_report(
        input_csv: Path = typer.Option("data/processed/test_results.csv", help="Input CSV with scoring results"),
        output_html: Path = typer.Option(None, help="Output HTML report path (optional)"),
        output_csv: Path = typer.Option(None, help="Output CSV metrics path (optional)")
    ):
        """Generate risk report and export metrics"""
        try:
            results_df = pd.read_csv(input_csv)
            logger.info(f"✓ Loaded {len(results_df)} scoring results")
            
            reporter = RiskReporter()
            reporter.generate_html_report(results_df, output_html)
            reporter.export_metrics_csv(results_df, output_csv)
            
            logger.success("✓ Risk report generation complete!")
        except FileNotFoundError as e:
            logger.error(f"✗ File not found: {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            logger.error(f"✗ Report generation failed: {e}")
            raise typer.Exit(code=1)
    
    app()
    
if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def generate_report(input_csv: str):
        results_df = pd.read_csv(input_csv)
        reporter = RiskReporter()
        reporter.generate_html_report(results_df)
        print("✓ Report generated!")
    
    app()