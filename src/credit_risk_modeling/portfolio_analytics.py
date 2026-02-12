"""Portfolio analytics and risk aggregation"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio statistics"""
    total_applicants: int
    approval_rate: float
    denial_rate: float
    manual_review_rate: float
    avg_pd: float
    avg_lgd: float
    avg_ead: float
    total_ead: float
    total_expected_loss: float
    
    # Segmentation
    by_risk_tier: dict
    by_loan_intent: dict
    by_decision: dict

class PortfolioAnalyzer:
    """Analyze aggregated portfolio risk metrics"""
    
    def __init__(self):
        logger.info("✓ PortfolioAnalyzer initialized")
    
    def analyze(self, scoring_results_df: pd.DataFrame) -> PortfolioMetrics:
        """
        Compute portfolio-level metrics from batch scoring results.
        
        Args:
            scoring_results_df: DataFrame from batch_predict() output
                               Must have columns: decision, pd, lgd, ead, expected_loss,
                                                  risk_tier, loan_intent
        
        Returns:
            PortfolioMetrics dataclass with aggregated stats
        """
        df = scoring_results_df.copy()
        
        # Overall metrics
        total_applicants = len(df)
        approval_rate = (df['decision'] == 'APPROVE').sum() / total_applicants
        denial_rate = (df['decision'] == 'DENY').sum() / total_applicants
        manual_review_rate = (df['decision'] == 'MANUAL_REVIEW').sum() / total_applicants
        
        avg_pd = df['pd'].mean()
        avg_lgd = df['lgd'].mean()
        avg_ead = df['ead'].mean()
        total_ead = df['ead'].sum()
        total_expected_loss = df['expected_loss'].sum()
        
        # By risk tier
        by_risk_tier = {
            'LOW': self._tier_stats(df[df['risk_tier'] == 'LOW']),
            'MEDIUM': self._tier_stats(df[df['risk_tier'] == 'MEDIUM']),
            'HIGH': self._tier_stats(df[df['risk_tier'] == 'HIGH']),
        }
        
        # By loan intent
        by_loan_intent = {}
        for intent in df['loan_intent'].unique():
            by_loan_intent[intent] = self._tier_stats(df[df['loan_intent'] == intent])
        
        # By decision
        by_decision = {
            'APPROVE': self._tier_stats(df[df['decision'] == 'APPROVE']),
            'MANUAL_REVIEW': self._tier_stats(df[df['decision'] == 'MANUAL_REVIEW']),
            'DENY': self._tier_stats(df[df['decision'] == 'DENY']),
        }
        
        metrics = PortfolioMetrics(
            total_applicants=total_applicants,
            approval_rate=approval_rate,
            denial_rate=denial_rate,
            manual_review_rate=manual_review_rate,
            avg_pd=avg_pd,
            avg_lgd=avg_lgd,
            avg_ead=avg_ead,
            total_ead=total_ead,
            total_expected_loss=total_expected_loss,
            by_risk_tier=by_risk_tier,
            by_loan_intent=by_loan_intent,
            by_decision=by_decision
        )
        
        logger.info(f"✓ Portfolio analyzed: {total_applicants} applicants")
        logger.info(f"  Approval rate: {approval_rate*100:.1f}%")
        logger.info(f"  Total EAD: ${total_ead:,.0f}")
        logger.info(f"  Total Expected Loss: ${total_expected_loss:,.0f}")
        
        return metrics
    
    @staticmethod
    def _tier_stats(tier_df: pd.DataFrame) -> dict:
        """Compute stats for a segment/tier"""
        if len(tier_df) == 0:
            return {
                'count': 0,
                'avg_pd': np.nan,
                'avg_lgd': np.nan,
                'avg_ead': np.nan,
                'total_ead': 0,
                'total_expected_loss': 0,
                'approval_rate': 0
            }
        
        return {
            'count': len(tier_df),
            'avg_pd': tier_df['pd'].mean(),
            'avg_lgd': tier_df['lgd'].mean(),
            'avg_ead': tier_df['ead'].mean(),
            'total_ead': tier_df['ead'].sum(),
            'total_expected_loss': tier_df['expected_loss'].sum(),
            'approval_rate': (tier_df['decision'] == 'APPROVE').sum() / len(tier_df) if len(tier_df) > 0 else 0
        }
    
    def report(self, metrics: PortfolioMetrics) -> str:
        """Generate human-readable portfolio report"""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║               PORTFOLIO RISK REPORT                           ║
╚════════════════════════════════════════════════════════════════╝

OVERVIEW
────────────────────────────────────────────────────────────────
Total Applicants:        {metrics.total_applicants:>10}
Approval Rate:           {metrics.approval_rate*100:>9.1f}%
Manual Review Rate:      {metrics.manual_review_rate*100:>9.1f}%
Denial Rate:             {metrics.denial_rate*100:>9.1f}%

RISK METRICS
────────────────────────────────────────────────────────────────
Average PD:              {metrics.avg_pd:>9.2%}
Average LGD:             {metrics.avg_lgd:>9.2%}
Average EAD:             ${metrics.avg_ead:>10,.0f}
Total EAD:               ${metrics.total_ead:>10,.0f}
Total Expected Loss:     ${metrics.total_expected_loss:>10,.0f}

BY RISK TIER
────────────────────────────────────────────────────────────────
"""
        for tier in ['LOW', 'MEDIUM', 'HIGH']:
            stats = metrics.by_risk_tier[tier]
            if stats['count'] > 0:
                report += f"""
{tier} Risk Tier:
  Count:                 {stats['count']:>10}
  Avg PD:                {stats['avg_pd']:>9.2%}
  Avg EAD:               ${stats['avg_ead']:>10,.0f}
  Total EAD:             ${stats['total_ead']:>10,.0f}
  Total Expected Loss:   ${stats['total_expected_loss']:>10,.0f}
"""
        
        report += "\nBY LOAN INTENT\n────────────────────────────────────────────────────────────────\n"
        for intent, stats in metrics.by_loan_intent.items():
            if stats['count'] > 0:
                report += f"""
{intent}:
  Count:                 {stats['count']:>10}
  Approval Rate:         {stats['approval_rate']*100:>9.1f}%
  Avg PD:                {stats['avg_pd']:>9.2%}
  Total EAD:             ${stats['total_ead']:>10,.0f}
"""
        
        report += "\nBY DECISION\n────────────────────────────────────────────────────────────────\n"
        for decision, stats in metrics.by_decision.items():
            if stats['count'] > 0:
                report += f"""
{decision}:
  Count:                 {stats['count']:>10}
  Avg PD:                {stats['avg_pd']:>9.2%}
  Total EAD:             ${stats['total_ead']:>10,.0f}
  Total Expected Loss:   ${stats['total_expected_loss']:>10,.0f}
"""
        
        report += "\n" + "="*66 + "\n"
        return report
    
    def to_dataframe(self, metrics: PortfolioMetrics) -> pd.DataFrame:
        """Convert metrics to structured DataFrame for export"""
        summary_data = {
            'Metric': [
                'Total Applicants', 'Approval Rate (%)', 'Manual Review Rate (%)', 'Denial Rate (%)',
                'Average PD (%)', 'Average LGD (%)', 'Average EAD ($)', 'Total EAD ($)', 'Total Expected Loss ($)'
            ],
            'Value': [
                metrics.total_applicants,
                metrics.approval_rate * 100,
                metrics.manual_review_rate * 100,
                metrics.denial_rate * 100,
                metrics.avg_pd * 100,
                metrics.avg_lgd * 100,
                metrics.avg_ead,
                metrics.total_ead,
                metrics.total_expected_loss
            ]
        }
        
        return pd.DataFrame(summary_data)