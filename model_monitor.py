"""Model Monitoring: Detect performance drift and data quality issues"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger
from credit_risk_modeling.config import MODELS_DIR


@dataclass
class MonitoringMetrics:
    """Track model performance over time"""
    timestamp: str
    total_predictions: int
    approval_rate: float
    avg_pd: float
    avg_confidence: float
    
    # Data quality checks
    missing_values_pct: float
    outlier_count: int
    
    # Drift indicators (compared to baseline)
    approval_rate_drift: float = 0.0
    pd_drift: float = 0.0
    confidence_drift: float = 0.0
    
    # Alerts
    alerts: list = field(default_factory=list)


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, baseline_metrics: dict = None):
        """
        Initialize monitor with baseline metrics.
        
        Args:
            baseline_metrics: Dict with 'approval_rate', 'avg_pd', 'avg_confidence'
                             If None, uses defaults
        """
        self.baseline = baseline_metrics or {
            'approval_rate': 0.65,
            'avg_pd': 0.15,
            'avg_confidence': 0.85
        }
        logger.info(f"✓ ModelMonitor initialized with baseline: {self.baseline}")
    
    def check_data_quality(self, df: pd.DataFrame) -> tuple:
        """
        Check for data quality issues.
        
        Returns:
            (missing_pct: float, outlier_count: int)
        """
        # Missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        
        # Outliers (using IQR method on numeric columns)
        outlier_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_count += outliers
        
        return missing_pct, outlier_count
    
    def detect_drift(self, current_metrics: dict) -> list:
        """
        Detect significant drift from baseline.
        
        Args:
            current_metrics: Dict with current performance metrics
        
        Returns:
            List of alert strings
        """
        alerts = []
        
        # Approval rate drift > 5%
        approval_drift = abs(current_metrics['approval_rate'] - self.baseline['approval_rate'])
        if approval_drift > 0.05:
            alerts.append(
                f"⚠️  APPROVAL RATE DRIFT: {current_metrics['approval_rate']*100:.1f}% "
                f"(baseline: {self.baseline['approval_rate']*100:.1f}%, "
                f"diff: {approval_drift*100:+.1f}%)"
            )
        
        # PD drift > 20% relative change
        pd_drift = abs(current_metrics['avg_pd'] - self.baseline['avg_pd']) / self.baseline['avg_pd']
        if pd_drift > 0.20:
            alerts.append(
                f"⚠️  PD DRIFT: {current_metrics['avg_pd']:.3f} "
                f"(baseline: {self.baseline['avg_pd']:.3f}, "
                f"relative change: {pd_drift*100:+.1f}%)"
            )
        
        # Confidence drop > 10%
        confidence_drift = self.baseline['avg_confidence'] - current_metrics['avg_confidence']
        if confidence_drift > 0.10:
            alerts.append(
                f"⚠️  MODEL CONFIDENCE DROP: {current_metrics['avg_confidence']:.3f} "
                f"(baseline: {self.baseline['avg_confidence']:.3f}, "
                f"drop: {confidence_drift:.3f})"
            )
        
        return alerts
    
    def monitor_batch(self, scoring_results_df: pd.DataFrame) -> MonitoringMetrics:
        """
        Run full monitoring check on batch scoring results.
        
        Args:
            scoring_results_df: DataFrame from batch_predict()
        
        Returns:
            MonitoringMetrics dataclass with all checks
        """
        df = scoring_results_df.copy()
        
        # Calculate performance metrics
        metrics = {
            'approval_rate': (df['decision'] == 'APPROVE').sum() / len(df),
            'avg_pd': df['pd'].mean(),
            'avg_confidence': df['confidence'].mean()
        }
        
        # Data quality checks
        missing_pct, outlier_count = self.check_data_quality(df)
        
        # Drift detection
        alerts = self.detect_drift(metrics)
        
        # Additional quality alerts
        if missing_pct > 0.5:
            alerts.append(f"⚠️  HIGH MISSING DATA: {missing_pct:.2f}%")
        
        if outlier_count > len(df) * 0.05:
            alerts.append(f"⚠️  HIGH OUTLIER COUNT: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
        
        # Create monitoring record
        monitoring = MonitoringMetrics(
            timestamp=datetime.now().isoformat(),
            total_predictions=len(df),
            approval_rate=metrics['approval_rate'],
            avg_pd=metrics['avg_pd'],
            avg_confidence=metrics['avg_confidence'],
            missing_values_pct=missing_pct,
            outlier_count=outlier_count,
            approval_rate_drift=abs(metrics['approval_rate'] - self.baseline['approval_rate']),
            pd_drift=abs(metrics['avg_pd'] - self.baseline['avg_pd']),
            confidence_drift=abs(metrics['avg_confidence'] - self.baseline['avg_confidence']),
            alerts=alerts
        )
        
        # Log results
        logger.info(f"✓ Monitoring complete: {len(df)} predictions")
        logger.info(f"  Approval rate: {metrics['approval_rate']*100:.1f}%")
        logger.info(f"  Avg PD: {metrics['avg_pd']:.3f}")
        logger.info(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        
        if alerts:
            logger.warning(f"⚠️  {len(alerts)} alerts detected:")
            for alert in alerts:
                logger.warning(f"    {alert}")
        else:
            logger.success("✓ No significant drift detected")
        
        return monitoring


def save_monitoring_log(monitoring: MonitoringMetrics, path: Path = None):
    """Save monitoring record to CSV for audit trail."""
    if path is None:
        path = MODELS_DIR / "monitoring_log.csv"
    
    record_df = pd.DataFrame([{
        'timestamp': monitoring.timestamp,
        'total_predictions': monitoring.total_predictions,
        'approval_rate': monitoring.approval_rate,
        'avg_pd': monitoring.avg_pd,
        'avg_confidence': monitoring.avg_confidence,
        'missing_pct': monitoring.missing_values_pct,
        'outlier_count': monitoring.outlier_count,
        'alerts_count': len(monitoring.alerts)
    }])
    
    if path.exists():
        existing = pd.read_csv(path)
        record_df = pd.concat([existing, record_df], ignore_index=True)
    
    record_df.to_csv(path, index=False)
    logger.info(f"✓ Monitoring log saved to {path}")